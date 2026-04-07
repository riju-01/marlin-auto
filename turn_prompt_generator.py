"""
Adaptive Turn 2/3 prompt generation.

Compares model diffs against the PR change checklist to decide:
  - Which checklist items were completed
  - Which are still missing
  - Whether to push new items or re-prompt for gaps

This drives the models toward producing output that closely matches
the golden PR answer across 3 turns.
"""

import json

from humanizer import humanize_prompt
from llm_client import generate as llm_generate


def generate_turn_prompt(turn_num: int, prev_diffs: str, prev_prompts: list[str],
                         acceptance_criteria: str, api_key: str,
                         checklist: list[dict] = None,
                         completed_items: list[int] = None) -> tuple[str | None, list[int]]:
    """
    Generate an adaptive follow-up prompt for Turn 2 or Turn 3.

    Returns (prompt_text, updated_completed_items).
    The caller should track completed_items across turns.
    """
    if checklist is None:
        checklist = []
    if completed_items is None:
        completed_items = []

    # Step 1: Analyze which checklist items the models completed
    newly_completed = _analyze_completion(
        prev_diffs, checklist, completed_items, api_key
    )
    all_completed = sorted(set(completed_items + newly_completed))
    remaining = [item for item in checklist if item["id"] not in all_completed]

    # Step 2: Decide strategy
    turn1_target_count = min(5, len(checklist))
    turn1_items = [i for i in checklist if i["id"] <= turn1_target_count]
    turn1_done = all(i["id"] in all_completed for i in turn1_items)

    if turn_num == 2:
        if turn1_done and remaining:
            next_items = remaining[:2]
            strategy = "advance"
        elif not turn1_done:
            missed = [i for i in turn1_items if i["id"] not in all_completed]
            next_items = missed
            strategy = "retry"
        else:
            next_items = remaining[:2] if remaining else []
            strategy = "advance"
    else:  # Turn 3
        next_items = remaining
        strategy = "final"

    # Step 3: Generate the prompt
    prompt = _build_adaptive_prompt(
        turn_num, strategy, next_items, all_completed,
        checklist, prev_diffs, prev_prompts, acceptance_criteria, api_key
    )

    if prompt:
        prompt = humanize_prompt(prompt, api_key)

    return prompt, all_completed


def _analyze_completion(diffs: str, checklist: list[dict],
                        already_completed: list[int], api_key: str) -> list[int]:
    """Ask the LLM which checklist items are visible in the diffs."""
    if not checklist or not diffs:
        return []

    unchecked = [i for i in checklist if i["id"] not in already_completed]
    if not unchecked:
        return []

    items_desc = "\n".join(
        f"  {i['id']}. {i['description']} (files: {', '.join(i.get('files', []))})"
        for i in unchecked
    )

    prompt = f"""Analyze these code diffs and determine which checklist items were completed.

CHECKLIST (unchecked items only):
{items_desc}

DIFFS:
{diffs[:12000]}

For each item, decide if the diffs show that item was addressed.
Return ONLY a JSON array of completed item IDs, e.g. [1, 3, 5]
If none were completed, return []"""

    result = llm_generate(prompt, api_key=api_key)
    if not result:
        return []

    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1] if "\n" in result else result[3:]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]

    try:
        ids = json.loads(result.strip())
        if isinstance(ids, list):
            return [i for i in ids if isinstance(i, int)]
    except json.JSONDecodeError:
        pass
    return []


def _build_adaptive_prompt(turn_num: int, strategy: str, next_items: list[dict],
                           completed: list[int], checklist: list[dict],
                           prev_diffs: str, prev_prompts: list[str],
                           acceptance_criteria: str, api_key: str) -> str | None:
    """Generate the actual prompt text based on the adaptive strategy."""
    prev_prompt_text = "\n---\n".join(
        f"Turn {i+1} prompt:\n{p}" for i, p in enumerate(prev_prompts)
    )

    items_hint = "\n".join(
        f"- {item['description']}" for item in next_items
    ) if next_items else "(cleanup and verification)"

    completed_summary = ", ".join(str(c) for c in completed) if completed else "none yet"

    if strategy == "retry":
        strategy_instruction = (
            f"The models did NOT fully complete items from Turn 1. "
            f"Completed so far: [{completed_summary}]. "
            f"Write a prompt that gives STRONGER hints about the missing items below, "
            f"without giving away the exact solution. Be more specific about what files "
            f"and functions need changes. The model needs a nudge, not the answer."
        )
    elif strategy == "advance":
        strategy_instruction = (
            f"Turn 1 items are done (completed: [{completed_summary}]). "
            f"Now introduce the NEXT set of changes. Describe them as new issues "
            f"you noticed while reviewing the Turn 1 work. Keep it natural - "
            f"'while looking at the changes so far, I noticed we also need...'"
        )
    else:  # final
        strategy_instruction = (
            f"This is the FINAL turn. Completed so far: [{completed_summary}]. "
            f"Push for the remaining items below plus any cleanup, test fixes, "
            f"or polish needed. Frame it as 'wrapping up' - fix loose ends, "
            f"run tests, make sure everything is consistent."
        )

    system = f"""You are generating a Turn {turn_num} follow-up prompt for a coding evaluation.

Strategy: {strategy.upper()}
{strategy_instruction}

Items to address this turn:
{items_hint}

Rules:
1. 80-180 words
2. Describe the PROBLEM, not the solution
3. Be vague enough that the model discovers the approach, but specific enough it targets the right files
4. No PR references, no "act as", no line numbers
5. Drop apostrophes: dont, its, wont, doesnt
6. No em dashes
7. No trailing period on last sentence
8. Write like a developer leaving a code review comment or follow-up task
9. Do NOT repeat anything from previous prompts

Return ONLY the prompt text."""

    user_prompt = f"""Previous prompts (do NOT repeat):
{prev_prompt_text}

Acceptance criteria:
{acceptance_criteria[:2000]}

Latest diffs (what was done so far):
{prev_diffs[:8000]}

Generate the Turn {turn_num} prompt."""

    return llm_generate(system + "\n\n" + user_prompt, api_key=api_key)
