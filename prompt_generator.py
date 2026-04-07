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

from humanizer import humanize_prompt
from llm_client import generate as llm_generate


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
- Describe the PROBLEM, not the solution
- Target 150-250 words
- Write like a developer filing a GitHub issue - casual but clear
- Include what the codebase currently does wrong and what "done" looks like
- Reference general areas of the codebase (modules, packages) but dont micromanage
- Mix sentence lengths - some short (5-8 words), some longer (15-25 words)
- Use abbreviations: param, repo, config, deps
- No trailing period on the last sentence"""


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

Write a 150-250 word prompt that describes the PROBLEM these items solve.
Be VAGUE enough that the model has to discover the solution, but specific enough that a good model will address all {len(turn1_items)} items.
Do NOT describe the solution. Do NOT list the items explicitly. Instead, describe the symptoms and what "done" looks like.
The model should naturally arrive at solving items 1-{len(turn1_items)} if it understands the problem well.

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
