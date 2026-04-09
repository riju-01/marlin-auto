"""
Generate the full Phase 2 analysis document (phase2.md) from PR context.

This includes:
  - Repo definition (what the repo does)
  - PR definition (what this PR changes)
  - Edge cases (4-6 concrete cases)
  - Acceptance criteria (5-7 done-when items)
  - PR change checklist (6-8 concrete things the PR does)
  - Turn 1 prompt (150-250 words, covers items 1-5 vaguely)

Also generates standalone turn1_prompt.txt for copy-paste into HFI.
"""

import json

from pipeline.humanizer import humanize_prompt
from core.llm_client import generate as llm_generate


PHASE2_SYSTEM = """You are analyzing a GitHub PR for a coding model evaluation task.
Generate a structured analysis document.

CRITICAL RULES:
- Do NOT mention PR numbers, branch names, or "this PR" anywhere
- Do NOT use role-based prompting ("you are a senior engineer", "act as")
- Drop apostrophes: use dont, its, wont, doesnt, cant
- No em dashes or double hyphens
- No trailing period on the last sentence of each section
- Be specific about file names, function names, and module names from the diff
- Write like a real developer, not a formal report"""


PROMPT_RULES = """Additional rules for the Initial Prompt section ONLY:

STRUCTURE (rubric requires this):
- Start with a 1-2 sentence summary of what the task is before any detail
- Organize into short labeled sections. Use bold labels like **Where to work:** or **What to remove:** - NOT formal "Objective / Scope / Deliverables" headers
- Use bullet points to enumerate specific targets (APIs, methods, classes, paths)
- End with a **Done when:** section: 2-3 bullets that are testable completion criteria
- Keep total length 150-300 words. Structure doesnt mean verbose

TONE:
- Write like a developer posting in a team channel: casual but organized
- Tone: senior engineer who typed this in 3 minutes. Not angry, not formal. Just clear
- Drop apostrophes naturally: dont, its, wont, doesnt, cant, shouldnt
- Mix sentence lengths in the prose parts. Bullets can be terse
- Use abbreviations where natural: param, repo, config, deps, e.g.
- One or two natural asides are fine (e.g. "they had warnings")

CONTENT:
- Describe the PROBLEM, not the solution
- Reference general areas (modules, packages, config names) but be VAGUE about exact fixes
- The opening summary says what the task is. The sections say where/what/done-when
- No trailing period on the last bullet or sentence

NEVER:
- NEVER use: "Ideally", "Currently", "Additionally", "Furthermore", "Notably"
- NEVER use "Done means X" or "Objective:" or "Success Criteria:" as headers
- NEVER start consecutive sentences or bullets the same way
- NEVER write more than 2-3 sentences in a row without a bullet list or section break
- Avoid slang, profanity, or overly emotional language"""


def generate_phase2_doc(pr_data: dict, diff_text: str, api_key: str,
                        status_callback=None) -> dict | None:
    """
    Generate the full Phase 2 analysis.
    Returns dict with keys: repo_def, pr_def, edge_cases, acceptance_criteria, prompt
    Or None on failure.
    """
    meta = pr_data["meta"]
    repo_info = pr_data.get("repo_info", {})
    files = pr_data.get("files", [])

    file_summary = "\n".join(
        f"  {f['filename']} (+{f['additions']}/-{f['deletions']})"
        for f in files[:30]
    )

    diff_preview = diff_text[:10000] if diff_text else "(no diff available)"

    context = f"""Repository: {pr_data['owner']}/{pr_data['repo']}
Description: {repo_info.get('description', 'N/A')}
Language: {repo_info.get('language', 'unknown')}
Stars: {repo_info.get('stars', 0)}

PR Title: {meta['title']}
Author: {meta.get('author', 'unknown')} | State: {meta.get('state', 'unknown')} | Merged: {meta.get('merged_at', 'N/A')}
Stats: {meta.get('changed_files', 0)} files | +{meta.get('additions', 0)} -{meta.get('deletions', 0)}
Base Commit: {meta.get('base_sha', 'N/A')}

PR Description:
{meta['body'][:2000]}

Changed files:
{file_summary}

Diff preview:
{diff_preview}"""

    # Generate all sections in one call for coherence
    if status_callback:
        status_callback("Generating Phase 2 analysis (repo, PR, edge cases, criteria)...")

    analysis_prompt = f"""{PHASE2_SYSTEM}

{context}

Generate the following sections. Use the exact headers shown. Be SPECIFIC - reference actual files, functions, and modules from the diff.

## Repo Definition
[3-5 sentences: what this repo does, its architecture, key abstractions. Write casually.]

## PR Definition
[3-5 sentences: what this specific change does and why, which components are affected. Dont mention "PR" or pull request - describe it as "this change" or "these updates".]

## Edge Cases
[4-6 numbered items. Each should reference a specific file or function from the diff and describe a concrete scenario that could break.]

## Acceptance Criteria
[5-7 numbered "Done when..." items. Each must be a specific, observable, testable outcome.]

Return ONLY the sections above with the exact headers."""

    analysis = llm_generate(analysis_prompt, api_key=api_key)
    if not analysis:
        return None

    # Extract checklist of concrete changes from the PR
    if status_callback:
        status_callback("Extracting PR change checklist (6-8 items)...")

    checklist_prompt = f"""{PHASE2_SYSTEM}

{context}

Analyze the diff carefully. List 6-8 CONCRETE things this change does - not vague goals, but specific code-level changes. Each item should be one atomic change a developer can verify.

Format as a JSON array of objects with "id" (1-8), "description" (one sentence), "files" (array of filenames touched), and "complexity" ("simple", "moderate", "complex").

Example:
[
  {{"id": 1, "description": "Add disableBackgroundKeepAlive field to IChatSessionStartOptions interface", "files": ["src/vs/workbench/contrib/chat/common/chatService.ts"], "complexity": "simple"}},
  {{"id": 2, "description": "Thread the new option through ChatModel constructor", "files": ["src/vs/workbench/contrib/chat/common/chatModel.ts"], "complexity": "moderate"}}
]

Return ONLY valid JSON, no markdown fences, no explanation."""

    checklist_raw = llm_generate(checklist_prompt, api_key=api_key)
    checklist = _parse_checklist(checklist_raw)

    # Generate Turn 1 prompt covering items 1-5 vaguely (problem-oriented)
    if status_callback:
        status_callback("Generating Turn 1 prompt (covering items 1-5)...")

    turn1_items = checklist[:5] if len(checklist) >= 5 else checklist
    turn1_item_hints = "\n".join(
        f"- {item['description']}" for item in turn1_items
    )

    prompt_gen = f"""{PHASE2_SYSTEM}

{PROMPT_RULES}

{context}

The PR addresses these specific issues (items 1-{len(turn1_items)}):
{turn1_item_hints}

Write a 150-300 word STRUCTURED prompt. Start with a 1-2 sentence summary, then use bold-labeled sections and bullet points.
Be VAGUE enough that the reader has to figure out the approach, but clear about what the problem IS.
Dont list the changes needed. Dont mention specific functions to fix. Describe the symptoms and what "fixed" looks like.
A good developer reading this should NATURALLY discover items 1-{len(turn1_items)} by understanding the problem well.

BAD example (wall of text, no structure, buried instructions):
"Currently, the allocation logic is spread across several files. Ideally, we want to reduce the number of separate allocations. This makes the code harder to follow for anyone new. The config helpers are also outdated. Done means faster creation and the tests pass."

BAD example (too formal, template-y headers):
"## Objective\nRemove deprecated APIs.\n## Scope\nThe client module.\n## Success Criteria\n- Build passes."

GOOD example (structured but human):
"Chat sessions are sticking around in the background even when nobody is using them. Mostly the local chat agent. Its eating resources, especially when someone spawns a bunch and forgets about them.

**Where this shows up:**
- Background session lifecycle in the chat view layer
- The global config toggle that controls keepalive (too coarse)

**What needs to happen:**
- Sessions created for quick/disposable use shouldnt persist after the view closes
- The keepalive setting needs a per-session override, not just the global flag

**Done when:**
- Quick chat sessions dont survive past their parent view
- Existing long-lived sessions still work as before
- Tests cover both paths"

Return ONLY the prompt text, nothing else."""

    raw_prompt = llm_generate(prompt_gen, api_key=api_key)
    if not raw_prompt:
        return None

    if status_callback:
        status_callback("Humanizing prompt...")

    prompt = humanize_prompt(raw_prompt, api_key)

    # Parse sections from the analysis
    sections = _parse_sections(analysis)

    return {
        "repo_def": sections.get("repo_definition", sections.get("repo definition", "")),
        "pr_def": sections.get("pr_definition", sections.get("pr definition", "")),
        "edge_cases": sections.get("edge_cases", sections.get("edge cases", "")),
        "acceptance_criteria": sections.get("acceptance_criteria", sections.get("acceptance criteria", "")),
        "checklist": checklist,
        "prompt": prompt,
        "raw_analysis": analysis,
    }


def _parse_checklist(raw: str) -> list[dict]:
    """Parse the checklist JSON from LLM output."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    return []


def _parse_sections(text: str) -> dict:
    """Parse ## headers into a dict."""
    import re
    sections = {}
    parts = re.split(r'^## ', text, flags=re.MULTILINE)
    for part in parts[1:]:
        lines = part.strip().split("\n", 1)
        if len(lines) >= 2:
            header = lines[0].strip().lower().replace(" ", "_")
            body = lines[1].strip()
            sections[header] = body
        elif lines:
            header = lines[0].strip().lower().replace(" ", "_")
            sections[header] = ""
    return sections


def format_phase2_md(pr_data: dict, phase2: dict) -> str:
    """Format the full phase2.md document."""
    meta = pr_data["meta"]

    checklist = phase2.get("checklist", [])
    checklist_md = ""
    if checklist:
        checklist_lines = []
        for item in checklist:
            files = ", ".join(item.get("files", []))
            checklist_lines.append(
                f"{item['id']}. [{item.get('complexity', '?')}] {item['description']}\n"
                f"   Files: {files}"
            )
        checklist_md = "\n".join(checklist_lines)
    else:
        checklist_md = "(no checklist generated)"

    return f"""# MARLIN V3 -- PHASE 2 PROMPT PREPARATION

**PR URL:** {pr_data['url']}
**Author:** {meta.get('author', 'unknown')} | **State:** {meta.get('state', 'unknown')} | **Merged:** {meta.get('merged_at', 'N/A')}
**Stats:** {meta.get('changed_files', 0)} files | +{meta.get('additions', 0)} -{meta.get('deletions', 0)}
**Base Commit:** {meta.get('base_sha', 'N/A')}

---

## Repo Definition
{phase2['repo_def']}

## PR Definition
{phase2['pr_def']}

## PR Change Checklist
{checklist_md}

## Turn Strategy
- **Turn 1**: Items 1-{min(5, len(checklist))} (described as problem, not solution)
- **Turn 2**: If 1-5 done → items {min(5, len(checklist))+1}-{min(7, len(checklist))}. If gaps → re-prompt for missing items
- **Turn 3**: Remaining items + cleanup/tests to match golden PR

## Edge Cases
{phase2['edge_cases']}

## Acceptance Criteria
{phase2['acceptance_criteria']}

## Initial Prompt
{phase2['prompt']}
"""


def generate_turn1_prompt(pr_data: dict, diff_text: str, api_key: str) -> str | None:
    """Legacy wrapper - generates just the prompt for backward compat."""
    result = generate_phase2_doc(pr_data, diff_text, api_key)
    if result:
        return result["prompt"]
    return None
