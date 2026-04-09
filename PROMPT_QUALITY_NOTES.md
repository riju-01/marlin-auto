# Prompt quality rubric failures (tooling backlog)

Notes from external prompt review (e.g. Marlin / HFI-style checks). Use this when improving `prompt_generator.py`, `humanize_prompt()`, and any post-generation validation.

---

## 1. Grammar — FAIL pattern

**What failed**

- Wrong contractions: `Its` vs `It's`, missing apostrophes in `arent`, `dont`, etc., when the rubric expects standard English.
- Missing punctuation (e.g. no space after period between sentences).
- Run-on sentences that reviewers want split for readability.

**Tool fixes to consider**

- Optional **mode**: `STRICT_GRAMMAR=true` or rubric profile that runs a pass **after** humanization to restore standard contractions and punctuation for prompts only (do not apply to HFI feedback answers if those intentionally drop apostrophes).
- Add a **prompt-specific** grammar lint step (rules + light LLM fix) before saving `turn1_prompt.txt`.
- Revisit `PROMPT_RULES` / `humanize_prompt`: conflicting goals today — feedback text wants informal contractions; **Turn 1 prompts** for rubric submission may need correct grammar.

---

## 2. Prompt clarity — FAIL pattern

**What failed**

- Prompt read as too conversational or vague (`some deprecated code hanging around`, `remove the cruft`).
- Missing concrete instructions: which deprecated methods/classes, what to replace with, exact modules, clear deliverables.

**Tool fixes to consider**

- After Turn 1 prompt generation, optional **second pass**: expand with checklist-derived hints (method signatures, package roots) without copying the whole golden diff.
- `PROMPT_RULES` currently bias toward vagueness for model discovery; add a **rubric-safe variant** that keeps problem-first tone but injects:
  - named packages / globs,
  - 2–4 example symbols (interfaces + overload patterns),
  - explicit "update tests" and "in-repo references" lines.
- Config flag: `--prompt-style rubric|discovery` (names TBD).

---

## 3. Prompt scope appropriateness — FAIL pattern

**What failed**

- Scope described as unbounded (`all deprecated methods`, `clients and core modules`) on a large repo.

**Tool fixes to consider**

- Always generate an explicit **boundary paragraph** from PR file list: `clients/src/main/java/org/apache/kafka/clients/**`, etc.
- Ban phrasing like "all deprecated" without a path constraint; replace with "deprecated API tied to files touched in this task" or checklist scope.
- Pull scope strings from `checklist.json` / changed files in `pr_fetcher` output.

---

## 4. Prompt technical specificity — FAIL pattern

**What failed**

- Mentioned `Duration` vs `TimeUnit` but not: test updates, backward compatibility, migration, deprecation annotations, version tags, docs.

**Tool fixes to consider**

- Template slots filled from Phase 2 sections: edge cases + acceptance criteria → bullet lines in the prompt (tests, migration expectation, build command if known).
- LLM instruction: "Include: test expectations, in-repo vs external breakage, and how deprecations are identified (@Deprecated, release notes) when relevant."

---

## Example task (Kafka PR 10438)

Original prompt failed all four areas. Manual fix that passed intent:

- Correct grammar and punctuation.
- Scoped directories under `clients/src/main/java/...`.
- Named patterns (`close(long, TimeUnit)`, `ConsumerConfig` / `ProducerConfig`, concrete types where helpful).
- Stated: update tests, in-tree references, external callers may break if they ignored warnings.
- Completion: build green, tests pass, no remaining references in source.

Reference files: `tasks/apache_kafka_10438/turn1_prompt.txt`, `phase2.md` (Initial Prompt section).

---

## Code touchpoints (for later)

| Area | File(s) |
|------|---------|
| Turn 1 wording rules | `prompt_generator.py` — `PROMPT_RULES`, `generate_phase2_doc()` |
| Post-process / informal tone | `humanizer.py` — `humanize_prompt()` |
| Checklist + file scope | `prompt_generator.py` checklist step; `pr_fetcher.py` |
| Categories for HFI survey | (planned) `prompt_generator.py` + `marlin_auto.py` pre-thread panel |

---

*Last updated from user-reported rubric feedback (grammar, clarity, scope, technical specificity).*
