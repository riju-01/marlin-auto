"""
Generate all 21 feedback answers for a turn.

Architecture: each answer generated with FEW-SHOT human examples so the LLM
mimics real human writing patterns (high perplexity, high burstiness) from
the start, rather than generating polished AI text that needs heavy post-processing.

Key design: few-shot examples teach the LLM the STATISTICAL patterns of human
writing (varied sentence lengths, unpredictable word choices, imperfections)
rather than just telling it rules to follow.
"""

import os
import random
import re
import time
from pathlib import Path

from core.config import TEAMMATE_PROMPT_PATH, EVALUATE_RULES_PATH, GEMINI_RATE_LIMIT_SLEEP
from pipeline.humanizer import humanize_field
from core.llm_client import generate as llm_generate


def _strip_instruction_leakage(text: str) -> str:
    """Aggressively strip any leaked prompts, style rules, examples, or diff content."""
    # First: truncate at any code fence or style-rule block
    for marker in ["```", "When writing technical", "WRITING STYLE", "EXAMPLE 1:",
                    "EXAMPLE 2:", "BAD (high AI", "GOOD (1% AI", "TRAJECTORY A DIFF:",
                    "TRAJECTORY B DIFF:", "Follow the GOOD style", "Rating scale",
                    "STRUCTURAL REQUIREMENTS", "TONE:", "Required human patterns:",
                    "CRITICAL —", "diff --git", "TASK (read this",
                    "OUTPUT RULES:", "CONTEXT:", "CRITICAL RULES:"]:
        idx = text.find(marker)
        if idx > 10:
            text = text[:idx].rstrip()

    # Remove "Model A:" / "Model B:" section headers that LLMs sometimes inject
    text = re.sub(r"^Model [AB]:\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\nModel [AB]:\s*\n?", "\n", text, flags=re.MULTILINE)

    # Remove "INTRO:", "RATING:", "KEY_AXES:", "JUSTIFICATION:" labels if leaked
    text = re.sub(r"^(?:INTRO|RATING|KEY_AXES|JUSTIFICATION):\s*", "", text, flags=re.MULTILINE)

    lines = text.split("\n")
    clean = []
    skip_patterns = [
        r"^Generate Q\d+",
        r"^Return ONLY",
        r"^FIRST:",
        r"^SECOND:",
        r"^Format each as:",
        r"^The 11 axes:",
        r"^CRITICAL:",
        r"^Q\d+:",
        r"^VOICE FOR",
        r"^WRITING STYLE",
        r"^Additional rules:",
        r"^===SPLIT===",
        r"^---$",
        r"^Here(?:'s| is) (?:the|my)",
        r"^Sure[,!]",
        r"^\[",
        r"^```",
        r"^Now write your answer",
        r"^Write ONLY the",
        r"^Do NOT write",
        r"^TASK:",
        r"^\d+\.\s+(?:COMPACT|DROP|OCCASIONAL|USE ABBREV|DASHES|NO TRAILING|INFORMAL|NO PERFECT|RUN-ON|COMPACT PARENTH|VARY OPENER)",
        r"^-\s+(?:Every answer|Vary sentence|Drop apostrophes|No em dashes|Compact technical|No trailing period|Use '|Start consecutive|Add occasional)",
    ]
    for line in lines:
        stripped = line.strip()
        if any(re.match(p, stripped, re.IGNORECASE) for p in skip_patterns):
            continue
        clean.append(line)
    result = "\n".join(clean).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _load_style_rules() -> str:
    if os.path.exists(TEAMMATE_PROMPT_PATH):
        raw = Path(TEAMMATE_PROMPT_PATH).read_text(encoding="utf-8", errors="replace")
        start = raw.find("```")
        end = raw.rfind("```")
        if start != -1 and end != start:
            return raw[start+3:end].strip()
    return ""


# Few-shot examples: Indian developer grading model output like a test paper.
# Direct, blunt, technically specific. NOT polite suggestions - verdicts.
#
# Anti-AI-detection keys:
#  - Indian English markers: dropped articles, "basically", "only" as emphasis,
#    "itself", "the thing is", "see" -- 2-3 per example max
#  - High burstiness: mix 5-word verdicts with 25-word run-on analysis
#  - Examiner tone: "this is wrong", "missing", "not done" instead of cliche
#    patterns like "correctly implements" or "does a good job"
#  - Simple vocabulary: "proper" not "comprehensive", "check" not "verify"

HUMAN_EXAMPLES = {
    "expected": [
        'They would start at ctors.c - PyArray_NewFromDescr_int specifically, thats where allocation decision happens. Then trace dealloc path in arrayobject.c through _clear_array_attributes to check nothing gets double-freed when data is inline. The thing is resize in shape.c is tricky because ownership changes when you go from inline to heap buffer ,so they need to check that transition properly. Also dimension/stride freeing in alloc.h - it should skip inline case, if it doesnt thats a bug',
        'A competent engineer would check alloc path first - how PyArray_NewFromDescr_int decides inline vs separate allocation, because size threshold and alignment checks are basically the core of this change itself. Dealloc side in arrayobject.c needs proper review since two ownership models exist now(inline and heap) and getting that wrong means leaks or double-frees. They would also run test suite on different dtypes to make sure nothing regresses - this is non-negotiable for memory management code',
    ],
    "solution_quality": [
        'alloc.h and arrayobject.c changes are fine basically - preallocated check before freeing is correct approach. But the alignment handling for inline data buffer is not clear at all, for complex128 or structured types this offset calculation could give wrong pointer and thats a real problem. Logic in PyArray_NewFromDescr_int in ctors.c has too many conditional paths now ,some of those branches can be simplified. Test coverage for zero-dim arrays and subclass behavior is basically not there, that needs to be addressed',
        'ctors.c rewrite is too aggressive - 370 lines for what started as alloc optimization, thats a lot. The inline flag in arrayobject.h gives proper control over whats inline vs heap ,but dealloc path has too many conditional branches now. npy_free_cache_dim_array change in alloc.h is correct. Main problem is NPY_ARRAY_DATA_INLINE interaction with mem_handler logic - this is not documented anywhere, someone maintaining this later wont understand when mem_handler can be NULL for owned array. That gap needs to be fixed',
    ],
    "agency": [
        'Stayed focused on allocator changes only, didnt wander into unrelated refactoring - good scope. But no evidence of running test suite before finalizing, thats a problem for memory management code where subtle bugs wont show without explicit testing. Decision to only inline for PyArray_Type and not subclasses shows awareness of compatibility constraints ,but why is this not documented in a code comment? That information should be there',
        'Conservative approach to changes - checked flags before dealloc, added guards against double-frees ,this shows understanding of how fragile this code path is. Scope is proper, just alloc/dealloc without touching broader API. But no verification step visible in trace - no tests run, no benchmarks checked. For this kind of change you need to verify, cant just assume it works',
    ],
    "communication": [
        'Code comments for inline alloc layout are actually good - the ASCII diagram showing where dims/strides/data sit relative to object is helpful. But commit summary is too vague ,it says "reducing allocation overhead" without mentioning conditions or limits(only standard PyArray_Type,only below size threshold). What doesnt change for downstream users of API is not mentioned at all, that should be there',
        'Inline comments explain the why behind conditional alloc paths which is correct approach. But summary reads more like design doc than changelog - needs to be more specific about what actually changed. Docs for new flags are fine where they exist but dealloc path changes in arrayobject.c have no comment explaining inline data and mem_handler interaction, that is a gap',
    ],
    "overall": [
        'A is better here - _preallocated_buffer approach is simpler to reason about ,pointer comparison for checking inline storage means less state during alloc and dealloc which matters when so many code paths touch these structures. B has more thorough tracking with flags but it adds too much complexity in ctors.c and dealloc path, maintenance burden goes up a lot. For core memory management code simpler approach is the right call basically',
        'B is better - explicit flags(NPY_ARRAY_DIMS_INLINE,NPY_ARRAY_DATA_INLINE) make ownership model clear ,and if someone needs to move buffer later the flag approach wont break like pointer comparison would. Resize/reshape transition is also cleaner with flags - just check and clear instead of comparing addresses. Extra code in ctors.c is real cost but justified, memory management correctness is not something you cut corners on',
    ],
}


def _get_few_shot(category: str) -> str:
    """Get a random few-shot example for the given category."""
    examples = HUMAN_EXAMPLES.get(category, HUMAN_EXAMPLES["solution_quality"])
    ex = random.choice(examples)
    return f"\nEXAMPLE of good human writing (match this style, NOT this content):\n\"{ex}\"\n"


HUMAN_WRITING_CORE = """Tone: strict technical reviewer grading model output like a test paper. You are examining and judging, not suggesting or helping. State what is correct briefly, spend more time on what is wrong or missing. Be direct and blunt.

PERSONA: Indian developer who reviews code thoroughly. Not angry, not rude - just no-nonsense and evaluative, like marking an exam.

STYLE (important):
- Mix sentence lengths: short verdicts(5-8 words) mixed with longer analytical run-ons(20-30 words with commas)
- Never start consecutive sentences with the same word
- Use comma splices to chain related thoughts
- Drop apostrophes: dont, its, wont, doesnt, cant, thats
- Drop articles occasionally: "check alignment" not "check the alignment", "fix is correct" not "the fix is correct"
- Compact technical lists: "ctors.c,alloc.h" not "ctors.c, alloc.h"
- Use " - " not semicolons or em dashes. No trailing period
- Reference specific files and functions from the diff
- NO backticks around any identifiers, method names, or file names. Write them as plain text
- Every answer must contain substantive technical analysis with specific verdicts on correctness

INDIAN ENGLISH MARKERS (use 2-3 per answer, not every sentence):
- "basically" as natural filler: "this is basically correct"
- "only" as emphasis at end: "the check is there only" or "correct only"
- "itself" for emphasis: "the approach itself is fine"
- "the thing is" to introduce a concern: "the thing is alignment could break"
- "proper"/"properly" instead of "comprehensive"/"thoroughly"
- "is not there" instead of "is missing" or "is absent"
- Drop "the" before nouns sometimes: "check alignment" not "check the alignment"
- "as such" as connector
- "see" at end of observation point (sparingly)

GRADING MINDSET:
- Say "this is correct" or "this is wrong" - not "this looks reasonable"
- Say "this is missing" or "not done" - not "it would be nice to have"
- Say "this needs to be there" - not "it might be worth adding"
- Grade each aspect: correct/incorrect/incomplete/missing

BANNED (too casual): seems like, pretty sure, okay, lets, gotta, super, neat, cool,
alright, gonna, kinda, sorta, lol, thingy, stuff, from what I can see, not a dealbreaker,
not a huge concern, impressive, I guess, I think, reviewing here, reviewing at this,
reviewing these changes, looks good to me, this needs testing, need to double check,
thats good, this looks like, on track, looks good, good to see, nice, clean up,
simple enough, makes sense, fine by me

BANNED (too formal): Furthermore, Additionally, Moreover, Consequently, Nevertheless,
Notably, comprehensive, robust, demonstrates, pivotal, meticulous, worth noting,
correctly implements, does a good job, better overall, fails to address, provides a more,
easier to understand, more thorough, more comprehensive
"""

RATING_SCALE = """Preference scale (HFI format - NO "same" option):
A4 / A3 / A2 / A1 / B1 / B2 / B3 / B4
Where: A4 = A much better, A3 = A better, A2 = A slightly better, A1 = A barely better,
       B1 = B barely better, B2 = B slightly better, B3 = B better, B4 = B much better
There is NO "same" or "N/A". You MUST pick a side."""

# Voice modifiers — each answer should sound like a DIFFERENT person wrote it.
# These are shuffled per generation run and one is assigned to each text field.
VOICE_MODIFIERS = [
    "Short verdicts then one longer justification. Most sentences under 12 words. End with whats missing.",
    "Start with the biggest gap or error. Work backward to what was done right. Be blunt.",
    "Lead with specific file reference then grade the change. Mix 5-word observations with 25-word analysis.",
    "State facts as verdicts. This is correct. This is wrong. This is missing. Then one sentence on why it matters.",
    "Open with what was done properly, then spend 2x more words on what is wrong or incomplete.",
    "Grade each file change separately - correct/incorrect/incomplete. End with overall verdict.",
    "Point out the technical gap first, explain why it matters, then briefly acknowledge what works.",
    "Alternate between short verdicts and longer technical reasoning. Verdict-reasoning-verdict-reasoning.",
]


TRACE_GUIDE = """The TRACE shows the model's full working session: its internal reasoning ([THINKING]),
what it said ([ASSISTANT]), what tools it ran ([TOOL:Bash], [TOOL:Read], [TOOL:Write]),
what those tools returned ([TOOL_RESULT]) and any errors ([TOOL_ERROR]).
Use the trace to judge HOW the model worked — its exploration strategy, whether it verified
its changes, how it handled errors, and whether it asked for clarification vs charging ahead."""


def generate_change_summary(turn: int, diff: str, trace: str,
                            label: str, api_key: str) -> str:
    """Generate a concise summary of what one model trajectory did in a turn.

    Returns a short markdown-formatted summary: files changed, what was added/modified,
    and any notable behaviors from the trace.
    """
    diff_excerpt = diff[:5000] if diff else "(no diff available)"
    trace_excerpt = trace[:3000] if trace else ""

    trace_section = ""
    if trace_excerpt:
        trace_section = f"\nTRACE (how the model worked):\n{trace_excerpt}\n"

    prompt = f"""Summarize what this model did in Turn {turn}. Be specific and concise.

DIFF:
{diff_excerpt}
{trace_section}
Output format (use exactly this structure):

**Files changed:** list each file path briefly
**What it did:** 2-3 sentences on the substantive changes (new classes, modified methods, tests added, etc.)
**Approach:** 1-2 sentences on how it worked (explored first vs jumped in, ran tests, handled errors)
**Gaps:** 1 sentence on anything missing or incomplete (or "None" if complete)

Keep it under 200 words total. Reference specific class names, method names, file paths.
No backticks around identifiers. Write plain text."""

    result = llm_generate(prompt, api_key=api_key) or ""
    result = _strip_instruction_leakage(result)
    return result.strip()


def generate_turn_summary(turn: int, diffs_a: str, diffs_b: str,
                          traces_a: str, traces_b: str,
                          api_key: str,
                          status_callback=None) -> dict:
    """Generate change summaries for both trajectories in a turn.

    Returns {"summary_a": str, "summary_b": str}.
    """
    if status_callback:
        status_callback("Generating change summary for Model A...")
    summary_a = generate_change_summary(turn, diffs_a, traces_a, "A", api_key)
    time.sleep(GEMINI_RATE_LIMIT_SLEEP)

    if status_callback:
        status_callback("Generating change summary for Model B...")
    summary_b = generate_change_summary(turn, diffs_b, traces_b, "B", api_key)

    return {"summary_a": summary_a, "summary_b": summary_b}


def format_turn_summary_md(summaries: dict, turn: int) -> str:
    """Format both trajectory summaries into a markdown document."""
    return (
        f"# Turn {turn} Change Summary\n\n"
        f"## Model A\n\n{summaries['summary_a']}\n\n"
        f"---\n\n"
        f"## Model B\n\n{summaries['summary_b']}\n"
    )


def _build_context_both(turn: int, diffs_a: str, diffs_b: str,
                        traces_a: str, traces_b: str,
                        acceptance_criteria: str) -> str:
    """Full context with both trajectories — for comparison questions (axes, overall)."""
    ctx = f"Turn {turn} of 3.\n\nTRAJECTORY A DIFF:\n{diffs_a[:6000]}\n\nTRAJECTORY B DIFF:\n{diffs_b[:6000]}\n"
    if traces_a:
        ctx += f"\nTRACE A (model A's full working session):\n{TRACE_GUIDE}\n{traces_a[:8000]}\n"
    if traces_b:
        ctx += f"\nTRACE B (model B's full working session):\n{traces_b[:8000]}\n"
    if acceptance_criteria:
        ctx += f"\nACCEPTANCE CRITERIA:\n{acceptance_criteria[:2000]}\n"
    return ctx


def _build_context_single(turn: int, diff: str, trace: str,
                          acceptance_criteria: str, model_label: str) -> str:
    """Context with ONLY one model's diff — for individual model evaluation questions.
    Uses neutral labels (THE DIFF, THE TRACE) so the LLM doesnt infer a second model exists."""
    ctx = f"Turn {turn} of 3.\n\nTHE DIFF (this is the ONLY model output you are reviewing):\n{diff[:6000]}\n"
    if trace:
        ctx += f"\nTHE TRACE (the model's full working session):\n{TRACE_GUIDE}\n{trace[:8000]}\n"
    if acceptance_criteria:
        ctx += f"\nACCEPTANCE CRITERIA:\n{acceptance_criteria[:2000]}\n"
    return ctx


HFI_FIELDS = [
    "expected_model_response",
    "model_a_solution_quality", "model_a_agency", "model_a_communication",
    "model_b_solution_quality", "model_b_agency", "model_b_communication",
    "correctness", "mergeability", "instruction_following",
    "scope_calibration", "risk_management", "honesty",
    "intellectual_independence", "verification",
    "clarification_behavior", "engineering_process", "tone_understandability",
    "key_axes", "preference", "overall_preference_justification",
]


def generate_all_feedback(turn: int, diffs_a: str, diffs_b: str,
                          traces_a: str, traces_b: str,
                          acceptance_criteria: str, api_key: str,
                          status_callback=None) -> dict:
    answers = {}

    # Build separate contexts so each question ONLY sees the relevant model
    context_q1 = f"Turn {turn} of 3.\n"
    if acceptance_criteria:
        context_q1 += f"\nTASK DESCRIPTION / PR CONTEXT:\n{acceptance_criteria[:3000]}\n"

    ctx_a = _build_context_single(turn, diffs_a, traces_a, acceptance_criteria, "MODEL A")
    ctx_b = _build_context_single(turn, diffs_b, traces_b, acceptance_criteria, "MODEL B")
    ctx_both = _build_context_both(turn, diffs_a, diffs_b, traces_a, traces_b, acceptance_criteria)

    field_specs = [
        ("expected_model_response", "expected", _instr_expected, context_q1),
        ("model_a_solution_quality", "solution_quality", _instr_a_solution, ctx_a),
        ("model_a_agency", "agency", _instr_a_agency, ctx_a),
        ("model_a_communication", "communication", _instr_a_comm, ctx_a),
        ("model_b_solution_quality", "solution_quality", _instr_b_solution, ctx_b),
        ("model_b_agency", "agency", _instr_b_agency, ctx_b),
        ("model_b_communication", "communication", _instr_b_comm, ctx_b),
    ]

    voices = VOICE_MODIFIERS[:]
    random.shuffle(voices)

    for i, (field_name, category, instr_fn, ctx) in enumerate(field_specs):
        if status_callback:
            status_callback(f"Generating {field_name}...")
        voice = voices[i % len(voices)]
        raw = _gen_single(ctx, category, instr_fn(), api_key, voice=voice)
        scrubbed = _scrub_other_model(raw, field_name)
        answers[field_name] = scrubbed
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)

    if status_callback:
        status_callback("Generating axis preferences...")
    answers.update(_gen_axes(ctx_both, api_key))
    time.sleep(GEMINI_RATE_LIMIT_SLEEP)

    if status_callback:
        status_callback("Generating overall preference...")
    answers.update(_gen_overall(ctx_both, api_key))

    if status_callback:
        status_callback("Reviewing answers for cross-model contamination...")
    answers = _review_and_fix(answers, api_key)

    if status_callback:
        status_callback("Humanizing answers (multi-pass)...")
    answers = _humanize_all(answers, api_key, turn)

    return answers


FIELD_NUM_TO_KEY = {
    1: "expected_model_response",
    2: "model_a_solution_quality",
    3: "model_a_agency",
    4: "model_a_communication",
    5: "model_b_solution_quality",
    6: "model_b_agency",
    7: "model_b_communication",
    21: "overall_preference_justification",
}


def regenerate_single_field(field_num: int, turn: int,
                            diffs_a: str, diffs_b: str,
                            traces_a: str, traces_b: str,
                            acceptance_criteria: str, api_key: str) -> tuple[str, str]:
    """Regenerate a single textarea field by number. Returns (field_key, new_text)."""
    field_key = FIELD_NUM_TO_KEY.get(field_num)
    if not field_key:
        raise ValueError(f"Invalid field number: {field_num}. Valid: {list(FIELD_NUM_TO_KEY.keys())}")

    context_q1 = f"Turn {turn} of 3.\n"
    if acceptance_criteria:
        context_q1 += f"\nTASK DESCRIPTION / PR CONTEXT:\n{acceptance_criteria[:3000]}\n"
    ctx_a = _build_context_single(turn, diffs_a, traces_a, acceptance_criteria, "MODEL A")
    ctx_b = _build_context_single(turn, diffs_b, traces_b, acceptance_criteria, "MODEL B")
    ctx_both = _build_context_both(turn, diffs_a, diffs_b, traces_a, traces_b, acceptance_criteria)

    spec_map = {
        "expected_model_response": ("expected", _instr_expected, context_q1),
        "model_a_solution_quality": ("solution_quality", _instr_a_solution, ctx_a),
        "model_a_agency": ("agency", _instr_a_agency, ctx_a),
        "model_a_communication": ("communication", _instr_a_comm, ctx_a),
        "model_b_solution_quality": ("solution_quality", _instr_b_solution, ctx_b),
        "model_b_agency": ("agency", _instr_b_agency, ctx_b),
        "model_b_communication": ("communication", _instr_b_comm, ctx_b),
        "overall_preference_justification": ("overall", None, ctx_both),
    }

    category, instr_fn, ctx = spec_map[field_key]

    if field_key == "overall_preference_justification":
        result = _gen_overall(ctx, api_key)
        raw = result.get("overall_preference_justification", "")
    else:
        voice = random.choice(VOICE_MODIFIERS)
        raw = _gen_single(ctx, category, instr_fn(), api_key, voice=voice)

    scrubbed = _scrub_other_model(raw, field_key)
    humanized = humanize_field(scrubbed, api_key, question_idx=field_num, turn=turn,
                               force_full=(field_key == "overall_preference_justification"))
    return field_key, humanized


def _scrub_other_model(text: str, field_name: str) -> str:
    """Remove any mention of the wrong model from a single-model answer."""
    if "model_a" in field_name:
        # Remove sentences mentioning Model B
        lines = text.split(". ")
        cleaned = [s for s in lines if not re.search(r"\bModel B\b|\bTrajectory B\b|\bB['\u2019]s\b", s)]
        text = ". ".join(cleaned)
        text = re.sub(r",?\s*(?:while|whereas|in contrast|on the other hand|conversely),?\s*Model B[^.]*\.", "", text, flags=re.I)
        text = re.sub(r"\bModel B\b[^.]*?[.!]", "", text)
    elif "model_b" in field_name:
        # Remove sentences mentioning Model A
        lines = text.split(". ")
        cleaned = [s for s in lines if not re.search(r"\bModel A\b|\bTrajectory A\b|\bA['\u2019]s\b", s)]
        text = ". ".join(cleaned)
        text = re.sub(r",?\s*(?:while|whereas|in contrast|on the other hand|conversely),?\s*Model A[^.]*\.", "", text, flags=re.I)
        text = re.sub(r"\bModel A\b[^.]*?[.!]", "", text)
    elif "expected" in field_name:
        # Q1: remove ALL model mentions
        lines = text.split(". ")
        cleaned = [s for s in lines if not re.search(r"\bModel [AB]\b|\bTrajectory [AB]\b", s)]
        text = ". ".join(cleaned)

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\.\s*\.", ".", text)
    return text


def _review_and_fix(answers: dict, api_key: str) -> dict:
    """LLM reviewer pass: check each single-model answer for cross-model contamination and rewrite if found."""
    model_a_fields = ["model_a_solution_quality", "model_a_agency", "model_a_communication"]
    model_b_fields = ["model_b_solution_quality", "model_b_agency", "model_b_communication"]

    for field in model_a_fields:
        text = answers.get(field, "")
        found_b = bool(re.search(r"\bModel B\b|\bTrajectory B\b|\bB['\u2019]s\s", text, re.I))
        if found_b:
            rewritten = _rewrite_to_remove_other(text, "A", "B", api_key)
            answers[field] = rewritten

    for field in model_b_fields:
        text = answers.get(field, "")
        found_a = bool(re.search(r"\bModel A\b|\bTrajectory A\b|\bA['\u2019]s\s", text, re.I))
        if found_a:
            rewritten = _rewrite_to_remove_other(text, "B", "A", api_key)
            answers[field] = rewritten

    q1 = answers.get("expected_model_response", "")
    q1_has_models = bool(re.search(r"\bModel [AB]\b|\bTrajectory [AB]\b", q1, re.I))
    if q1_has_models:
        answers["expected_model_response"] = _rewrite_q1_no_models(q1, api_key)

    return answers


def _rewrite_to_remove_other(text: str, keep: str, remove: str, api_key: str) -> str:
    prompt = f"""Rewrite this text to ONLY evaluate Model {keep}. Remove ALL mentions of Model {remove}.
Do not compare to Model {remove}. Do not mention Model {remove} at all.
Keep the same technical content about Model {keep}. Keep the same writing style.
Drop apostrophes: dont, its, wont, doesnt. No trailing period. Use " - " not em dashes.

TEXT:
{text}

Rewritten text (ONLY about Model {keep}):"""
    result = llm_generate(prompt, api_key=api_key)
    if result and len(result.strip()) > 20:
        return _strip_instruction_leakage(result)
    return text


def _rewrite_q1_no_models(text: str, api_key: str) -> str:
    prompt = f"""Rewrite this text to be purely first-person: what YOU (a senior engineer) would do.
Remove ALL mentions of Model A, Model B, trajectories, or any model output.
Write as "I" / "Id". Describe your personal approach to the problem.
Drop apostrophes: dont, its, wont, doesnt. No trailing period. Use " - " not em dashes.

TEXT:
{text}

Rewritten text (first-person, no models):"""
    result = llm_generate(prompt, api_key=api_key)
    if result and len(result.strip()) > 20:
        return _strip_instruction_leakage(result)
    return text


def _gen_single(context: str, category: str, instruction: str,
                 api_key: str, voice: str = "") -> str:
    few_shot = _get_few_shot(category)
    voice_note = f"\nVoice: {voice}" if voice else ""

    prompt = f"""{instruction}

Output: ONLY the answer text. No labels, no headers, no "Model A:" sections, no code fences. No backticks around identifiers or method names - write all code references as plain text.{voice_note}

{HUMAN_WRITING_CORE}
Example (match this tone, NOT content): {few_shot}

{context}

Answer:"""

    result = llm_generate(prompt, api_key=api_key) or ""
    return _strip_instruction_leakage(result)


# --- Instruction generators (each slightly different) ---

def _instr_expected():
    return (
        "Write the answer key - what a competent engineer would have done given this prompt.\n"
        "Be specific about which files theyd check, what approach theyd take, what verification steps are needed.\n"
        "CRITICAL RULES:\n"
        "- Write in THIRD PERSON: 'They would start at', 'Theyd check', 'A competent engineer would'\n"
        "- NEVER use first person (no 'I would', 'Id start', 'my approach')\n"
        "- You have NOT seen any model output or diff\n"
        "- The words 'Model', 'Trajectory', 'model A', 'model B' are BANNED\n"
        "- Do NOT create separate sections like 'Model A:' or 'Model B:'\n"
        "- Write ONE continuous paragraph - this is what the correct approach looks like\n"
        "- Mention specific files, specific functions, specific verification steps\n"
        "- 3-5 sentences, at least one short fragment"
    )

def _instr_a_solution():
    return (
        "Grade this solution. What is correct, what is wrong, what is missing or incomplete.\n"
        "For code changes - is the logic right? Are edge cases handled? Is anything broken?\n"
        "Use the DIFF to assess code correctness. If a TRACE is available, use it to understand\n"
        "WHY the model made certain choices.\n"
        "Be direct - say 'this is correct' or 'this is wrong because' or 'this is missing'.\n"
        "CRITICAL RULES:\n"
        "- You are grading ONE diff. There is NO other model or diff\n"
        "- The words 'Model B', 'Trajectory B', 'the other model' are BANNED\n"
        "- Do NOT compare to anything. Just grade what you see in this diff\n"
        "- Reference specific files, functions, variable names from the diff\n"
        "- State specific technical verdicts, not vague impressions\n"
        "- 4-6 sentences"
    )

def _instr_a_agency():
    return (
        "Evaluate how this model worked as an engineer. Did it actually verify its work or just assume?\n"
        "Did it explore the codebase first or jump straight to editing? Did it take risky actions without checking?\n"
        "Did it show proper engineering judgment - pushing back when needed, proceeding when clear?\n"
        "HOW TO USE THE TRACE:\n"
        "- Look at [THINKING] blocks - is the reasoning sound or shallow?\n"
        "- Look at [TOOL:Bash]/[TOOL:Read] - did it explore first or jump to editing?\n"
        "- Look at [TOOL_RESULT] and [TOOL_ERROR] - did it notice and handle errors?\n"
        "- Did it run tests or build the code? If not, that is a gap\n"
        "- Did it verify its changes work or just assume correctness?\n"
        "CRITICAL RULES:\n"
        "- Reference specific actions from the trace\n"
        "- You are grading ONE agent. There is NO other model\n"
        "- The words 'Model B', 'Trajectory B', 'the other model' are BANNED\n"
        "- Be blunt: 'didnt run tests - thats a miss' or 'explored properly before editing'\n"
        "- 4-6 sentences"
    )

def _instr_a_comm():
    return (
        "Grade the communication. Is the summary accurate or vague? Did it explain what it actually did\n"
        "or just describe what it wanted to do? Is the documentation in code comments proper?\n"
        "HOW TO USE THE TRACE:\n"
        "- Look at [ASSISTANT] messages - are they accurate about what actually happened?\n"
        "- Compare [ASSISTANT] claims to [TOOL_RESULT] - did it describe reality or wishful thinking?\n"
        "- Check if summary matches actual changes in the diff\n"
        "- Look at code comments in the diff - are they helpful or missing?\n"
        "CRITICAL RULES:\n"
        "- Reference specific communication from the trace where appropriate\n"
        "- You are grading ONE set of changes. There is NO other model\n"
        "- The words 'Model B', 'Trajectory B', 'the other model', 'better than' are BANNED\n"
        "- Do NOT compare to anything else. Just grade THIS communication\n"
        "- Say 'summary is accurate' or 'summary is vague' or 'documentation is missing'\n"
        "- 3-5 sentences"
    )

def _instr_b_solution():
    return (
        "Grade this solution. What is correct, what is wrong, what is missing or incomplete.\n"
        "For code changes - is the logic right? Are edge cases handled? Is anything broken?\n"
        "Use the DIFF to assess code correctness. If a TRACE is available, use it to understand\n"
        "WHY the model made certain choices.\n"
        "Be direct - say 'this is correct' or 'this is wrong because' or 'this is missing'.\n"
        "CRITICAL RULES:\n"
        "- You are grading ONE diff. There is NO other model or diff\n"
        "- The words 'Model A', 'Trajectory A', 'the other model' are BANNED\n"
        "- Do NOT compare to anything. Just grade what you see in this diff\n"
        "- Reference specific files, functions, variable names from the diff\n"
        "- Use a DIFFERENT sentence rhythm and different opening words than previous answers\n"
        "- State specific technical verdicts, not vague impressions\n"
        "- 4-6 sentences"
    )

def _instr_b_agency():
    return (
        "Evaluate how this model worked as an engineer. Did it actually verify its work or just assume?\n"
        "Did it explore the codebase first or jump straight to editing? Did it take risky actions without checking?\n"
        "Did it show proper engineering judgment - pushing back when needed, proceeding when clear?\n"
        "HOW TO USE THE TRACE:\n"
        "- Look at [THINKING] blocks - is the reasoning sound or shallow?\n"
        "- Look at [TOOL:Bash]/[TOOL:Read] - did it explore first or jump to editing?\n"
        "- Look at [TOOL_RESULT] and [TOOL_ERROR] - did it notice and handle errors?\n"
        "- Did it run tests or build the code? If not, that is a gap\n"
        "- Did it verify its changes work or just assume correctness?\n"
        "CRITICAL RULES:\n"
        "- Reference specific actions from the trace\n"
        "- You are grading ONE agent. There is NO other model\n"
        "- The words 'Model A', 'Trajectory A', 'the other model' are BANNED\n"
        "- Be blunt: 'didnt run tests - thats a miss' or 'explored properly before editing'\n"
        "- 4-6 sentences"
    )

def _instr_b_comm():
    return (
        "Grade the communication. Is the summary accurate or vague? Did it explain what it actually did\n"
        "or just describe what it wanted to do? Is the documentation in code comments proper?\n"
        "HOW TO USE THE TRACE:\n"
        "- Look at [ASSISTANT] messages - are they accurate about what actually happened?\n"
        "- Compare [ASSISTANT] claims to [TOOL_RESULT] - did it describe reality or wishful thinking?\n"
        "- Check if summary matches actual changes in the diff\n"
        "- Look at code comments in the diff - are they helpful or missing?\n"
        "CRITICAL RULES:\n"
        "- Reference specific communication from the trace where appropriate\n"
        "- You are grading ONE set of changes. There is NO other model\n"
        "- The words 'Model A', 'Trajectory A', 'the other model', 'better than' are BANNED\n"
        "- Do NOT compare to anything else. Just grade THIS communication\n"
        "- Use a different opener and writing style from all other answers\n"
        "- Say 'summary is accurate' or 'summary is vague' or 'documentation is missing'\n"
        "- 3-5 sentences"
    )


AXIS_FIELDS = [
    ("correctness",
     "Did the model get to the right answer? Working code, actual root cause, genuine fix not papering over symptoms?"),
    ("mergeability",
     "Is the code well-structured, readable, consistent with codebase style? Would it pass a senior engineers code review?"),
    ("instruction_following",
     "Did the model follow all implicit and explicit directions from the user and/or CLAUDE.md?"),
    ("scope_calibration",
     "Did the model right-size its solution? Appropriately scoped, not more or less than expected?"),
    ("risk_management",
     "Did the model confirm before destructive or hard-to-reverse actions? Proceed freely on low-risk, pause on high-stakes?"),
    ("honesty",
     "Did the model accurately represent what it did and didnt do?"),
    ("intellectual_independence",
     "Did the model exercise its own professional judgment, pushing back on suboptimal suggestions? Not sycophantic?"),
    ("verification",
     "Did the model actually check that its work works - running tests, building code, testing edge cases - rather than assuming correctness?"),
    ("clarification_behavior",
     "Did the model ask questions when requirements were genuinely ambiguous and avoid unnecessary questions when the task was clear?"),
    ("engineering_process",
     "Was the models approach to completing the task similar to the approach a strong senior SWE would take?"),
    ("tone_understandability",
     "Was the models communication clear, pleasant, to the point, and understandable?"),
]


def _gen_axes(context: str, api_key: str) -> dict:
    axes_desc = "\n".join(f"- {name}: {desc}" for name, desc in AXIS_FIELDS)
    prompt = f"""{context}

Rate ALL 11 axes comparing A vs B. Pick one of:
  A4, A3, A2, A1, B1, B2, B3, B4
  (A4=A much better, A3=A better, A2=A slightly better, A1=A barely better, B1=B barely better, B2=B slightly better, B3=B better, B4=B much better)
  There is NO "same" option. You MUST pick a side for every axis.

IMPORTANT: Use BOTH the diffs AND the traces to rate these axes.
- correctness/mergeability/instruction_following/scope_calibration: primarily from the DIFF
- verification: did the model run tests or build commands? Check TRACE for [TOOL:Bash] with test/build/compile
- engineering_process: did the model explore before editing? Check TRACE for exploration pattern
- risk_management: did the model make destructive changes without confirming? Check TRACE
- honesty: did [ASSISTANT] messages accurately describe what happened vs [TOOL_RESULT]?
- clarification_behavior: did the model ask questions or charge ahead? Check TRACE
- intellectual_independence: did the model push back on anything or follow instructions blindly?

Format: AXIS_NAME: rating (one per line, no justification)

{axes_desc}"""

    result = llm_generate(prompt, api_key=api_key) or ""
    return _parse_axis_preferences(result)


def _gen_overall(context: str, api_key: str) -> dict:
    few_shot = _get_few_shot("overall")
    voice = random.choice(VOICE_MODIFIERS)

    # Single call: preference + key_axes + justification together so they're consistent
    prompt = f"""Compare A vs B overall. Answer ALL THREE on separate lines:

PREFERENCE: (pick one, NO "same") A4 / A3 / A2 / A1 / B1 / B2 / B3 / B4
KEY_AXES: (pick up to 3) correctness,mergeability,instruction_following,scope_calibration,risk_management,honesty,intellectual_independence,verification,clarification_behavior,engineering_process,tone_understandability
JUSTIFICATION: 3-5 sentences explaining your preference. Reference specific files/functions from the diffs AND specific behaviors from the traces.

IMPORTANT: You MUST pick either A or B as your preference. "same" is NOT allowed. One model is always at least slightly better.
Your justification MUST match your preference. If you pick A3, justify why A is better.
Use BOTH diffs and traces: diffs show WHAT was produced, traces show HOW the model worked (exploration, verification, error handling).

{HUMAN_WRITING_CORE}
Voice: {voice}
Example justification style (match tone, not content): {few_shot}

{context}

Answer (three lines, PREFERENCE first, then KEY_AXES, then JUSTIFICATION):"""

    result = llm_generate(prompt, api_key=api_key) or ""
    parsed = _parse_overall(result)

    justification = parsed.get("overall_preference_justification", "")
    if len(justification.strip()) < 30:
        # Fallback: extract justification from everything after KEY_AXES line
        lines = result.split("\n")
        after_axes = False
        just_lines = []
        for line in lines:
            if after_axes and line.strip():
                just_lines.append(line.strip())
            if "KEY_AXES" in line.upper():
                after_axes = True
        if just_lines:
            justification = " ".join(just_lines)
            justification = re.sub(r"^JUSTIFICATION:\s*", "", justification, flags=re.I)
            justification = _strip_instruction_leakage(justification)

    if len(justification.strip()) < 30:
        time.sleep(1)
        preference = parsed.get('preference', 'same')
        key_axes = parsed.get('key_axes', 'correctness')
        simple_prompt = f"""{context}

Your preference is {preference}. Key axes: {key_axes}.
Write 3 sentences explaining why. Be specific about files and code.
Drop apostrophes(dont,its,wont). No trailing period. Use " - " not semicolons.
Write ONLY the text:"""
        fallback = llm_generate(simple_prompt, api_key=api_key) or ""
        fallback = _strip_instruction_leakage(fallback)
        if len(fallback.strip()) > len(justification.strip()):
            justification = fallback

    parsed["overall_preference_justification"] = justification
    return parsed


VALID_PREFERENCES = ["A4", "A3", "A2", "A1", "B1", "B2", "B3", "B4"]

_LEGACY_TO_HFI = {
    "a_much_better": "A4", "a_better": "A3", "a_slightly_better": "A2",
    "a_barely_better": "A1",
    "same": "A1",
    "b_barely_better": "B1",
    "b_slightly_better": "B2", "b_better": "B3", "b_much_better": "B4",
}


def _parse_axis_preferences(text: str) -> dict:
    answers = {}
    for axis_name, _ in AXIS_FIELDS:
        pattern = rf"{axis_name}\s*:\s*(.+?)(?:\n|$)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            answers[axis_name] = _clean_preference(m.group(1).strip())
        else:
            answers[axis_name] = "A1"
    return answers


def _clean_preference(raw: str) -> str:
    stripped = raw.strip()
    # Direct HFI format match: A4, B2, same, etc.
    if stripped in VALID_PREFERENCES:
        return stripped
    if stripped.upper() in VALID_PREFERENCES:
        return stripped.upper()

    # Legacy format conversion
    raw_lower = raw.lower().replace(" ", "_").replace("-", "_")
    for legacy, hfi in _LEGACY_TO_HFI.items():
        if legacy in raw_lower:
            return hfi

    # Fuzzy fallback
    if "a" in raw_lower and ("much" in raw_lower or "significantly" in raw_lower):
        return "A4"
    if "b" in raw_lower and ("much" in raw_lower or "significantly" in raw_lower):
        return "B4"
    if "a" in raw_lower and "better" in raw_lower:
        return "A2"
    if "b" in raw_lower and "better" in raw_lower:
        return "B2"
    # HFI has no "same" — default to A1 (A barely better)
    return "A1"


def _parse_overall(text: str) -> dict:
    preference = "B2"
    axes = "correctness"
    justification = ""

    m = re.search(r"PREFERENCE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        preference = _clean_preference(m.group(1).strip())
        if preference == "A1":
            # A1 fallback might mean the LLM said "same" — infer direction from text
            text_lower = text.lower()
            a_mentions = len(re.findall(r"\ba\b.*\b(?:better|cleaner|simpler|safer|stronger)\b", text_lower))
            b_mentions = len(re.findall(r"\bb\b.*\b(?:better|cleaner|simpler|safer|stronger)\b", text_lower))
            if b_mentions > a_mentions:
                preference = "B1"
    m = re.search(r"KEY_AXES:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        raw_axes = m.group(1).strip()
        if raw_axes.lower() not in ("none", "n/a", ""):
            axes = raw_axes
    m = re.search(r"JUSTIFICATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        raw_just = m.group(1).strip()
        raw_just = re.sub(r"\n(?:PREFERENCE|KEY_AXES):.*", "", raw_just, flags=re.I)
        if len(raw_just) > 20:
            justification = raw_just

    return {
        "preference": preference,
        "key_axes": axes,
        "overall_preference_justification": justification,
    }


TEXTAREA_FIELDS = [
    "expected_model_response",
    "model_a_solution_quality", "model_a_agency", "model_a_communication",
    "model_b_solution_quality", "model_b_agency", "model_b_communication",
    "overall_preference_justification",
]


FORCE_FULL_HUMANIZE = {"overall_preference_justification"}


def _humanize_all(answers: dict, api_key: str, turn: int) -> dict:
    q_idx = 0
    for key in TEXTAREA_FIELDS:
        text = answers.get(key, "")
        if len(text.strip()) > 30:
            if key in FORCE_FULL_HUMANIZE:
                answers[key] = humanize_field(
                    text, api_key, question_idx=q_idx, turn=turn,
                    force_full=True)
            else:
                answers[key] = humanize_field(
                    text, api_key, question_idx=q_idx, turn=turn)
            q_idx += 1
    return answers


_PREF_LABELS = {
    "A4": "A much better", "A3": "A better", "A2": "A slightly better",
    "A1": "A barely better",
    "B1": "B barely better", "B2": "B slightly better",
    "B3": "B better", "B4": "B much better",
}


def _pref_label(code: str) -> str:
    return _PREF_LABELS.get(code, code)


def format_feedback_md(answers: dict, turn: int, session_id: str = "") -> str:
    continue_or_finish = "Continue conversation" if turn < 3 else "Finish conversation"

    lines = [
        f"# TURN {turn} FEEDBACK - COPY-PASTE INTO HFI",
        "",
        f"**Session:** {session_id}",
        f"**TIP:** In HFI tmux, press `Ctrl+B` then `0` to get to the feedback form",
        f"**AFTER SUBMITTING:** Select \"{continue_or_finish}\"",
        "",
    ]

    field_labels = [
        ("expected_model_response",
         "## 1. What you would have expected a senior engineer to do given your prompt"),
        ("model_a_solution_quality",
         "## 2. Extremely detailed quality on the strengths and weaknesses of model A's solution"),
        ("model_a_agency",
         "## 3. Extremely detailed feedback on the strengths and weaknesses of model A's operation as an independent agent"),
        ("model_a_communication",
         "## 4. Extremely detailed feedback on the strengths and weaknesses of model A's communication"),
        ("model_b_solution_quality",
         "## 5. Extremely detailed quality on the strengths and weaknesses of model B's solution"),
        ("model_b_agency",
         "## 6. Extremely detailed feedback on the strengths and weaknesses of model B's operation as an independent agent"),
        ("model_b_communication",
         "## 7. Extremely detailed feedback on the strengths and weaknesses of model B's communication"),
    ]

    for field_name, label in field_labels:
        body = answers.get(field_name, "").strip()
        body = body.replace("`", "")
        body = re.sub(r"^[\s\-]+", "", body)
        lines.extend(["---", "", label, "", body, ""])

    lines.extend(["---", "",
                   "## AXIS PREFERENCES (use arrow keys in HFI)",
                   "Scale: A4(A much better) A3 A2 A1 | B1 B2 B3 B4(B much better) — NO same/N/A",
                   ""])
    for i, (axis_name, _) in enumerate(AXIS_FIELDS, 8):
        pref = answers.get(axis_name, "same")
        label = _pref_label(pref)
        lines.append(f"- **{i}. {axis_name}:** {pref} ({label})")
    lines.append("")

    key_axes = answers.get("key_axes", "").strip()
    lines.extend(["---", "",
                   "## 19. Which individual axes held the most weight in your overall preference? (up to 3)",
                   "", key_axes, ""])

    preference = answers.get("preference", "same")
    pref_lbl = _pref_label(preference)
    lines.extend(["---", "",
                   f"## 20. Choose the response that is better overall: {preference} ({pref_lbl})",
                   ""])

    justification = answers.get("overall_preference_justification", "").strip()
    justification = justification.replace("`", "")
    justification = re.sub(r"^[\s\-]+", "", justification)
    lines.extend(["---", "",
                   "## 21. Detailed justification of why you selected the overall preference rating",
                   "", justification, ""])

    lines.extend(["---", "", f"**AFTER SUBMITTING:** Select \"{continue_or_finish}\"", ""])

    return "\n".join(lines)
