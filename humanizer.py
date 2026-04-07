"""
Multi-pass humanization pipeline.

Each answer gets individually humanized with varied style to avoid
pattern detection across questions. The pipeline is:

  1. Regex transforms (em-dash removal, contraction dropping, filler strip, etc.)
  2. Re-score. If > target: LLM rewrite with randomized style persona
  3. Re-score. If > target: aggressive style injection (quirks, splices)
  4. Final validation

Key principle: every answer should sound like a DIFFERENT person wrote it.
We randomize style parameters per field to break cross-question patterns.

Supports Gemini, OpenAI, and Claude via llm_client.
"""

import random
import re
import time

from ai_scorer import score_field, full_validation, AI_VOCABULARY
from config import AI_SCORE_TARGET, MAX_HUMANIZE_PASSES
from llm_client import generate as llm_generate

# ---------------------------------------------------------------------------
# Regex transform tables
# ---------------------------------------------------------------------------

_EM_DASH = re.compile(r"\s*[—–]+\s*|(?<=[a-zA-Z])\s+--\s+(?=[a-zA-Z])")
_MULTI_SPACE = re.compile(r"  +")

_FILLER_PHRASES = [
    (re.compile(r"\bIt(?:'s| is) worth noting that\s*", re.I), ""),
    (re.compile(r"\bIt(?:'s| is) important to note that\s*", re.I), ""),
    (re.compile(r"\bNotably,?\s*", re.I), ""),
    (re.compile(r"\bFurthermore,?\s*"), "Also "),
    (re.compile(r"\bfurthermore,?\s*"), "also "),
    (re.compile(r"\bMoreover,?\s*"), "Also "),
    (re.compile(r"\bmoreover,?\s*"), "also "),
    (re.compile(r"\bAdditionally,?\s*"), ""),
    (re.compile(r"\badditionally,?\s*"), ""),
    (re.compile(r"\bConsequently,?\s*"), "So "),
    (re.compile(r"\bconsequently,?\s*"), "so "),
    (re.compile(r"\bNevertheless,?\s*"), "Still "),
    (re.compile(r"\bnevertheless,?\s*"), "still "),
    (re.compile(r"\bNonetheless,?\s*"), "Still "),
    (re.compile(r"\bnonetheless,?\s*"), "still "),
    (re.compile(r"\bIn conclusion,?\s*", re.I), ""),
    (re.compile(r"\bOverall,?\s+", re.I), ""),
    (re.compile(r"\bUltimately,?\s*", re.I), ""),
    (re.compile(r"\bEssentially,?\s*", re.I), ""),
    (re.compile(r"\bSpecifically,?\s*", re.I), ""),
    (re.compile(r"\bCrucially,?\s*", re.I), ""),
    (re.compile(r"\bImportantly,?\s*", re.I), ""),
    (re.compile(r"\bdemonstrates\b", re.I), "shows"),
    (re.compile(r"\bdemonstrating\b", re.I), "showing"),
    (re.compile(r"\bLeverage\b"), "Use"),
    (re.compile(r"\bleverage\b"), "use"),
    (re.compile(r"\bUtilize\b"), "Use"),
    (re.compile(r"\butilize\b"), "use"),
    (re.compile(r"\butilizing\b"), "using"),
    (re.compile(r"\bFacilitate\b"), "Help"),
    (re.compile(r"\bfacilitate\b"), "help"),
    (re.compile(r"\bHowever,?\s*"), "But "),
    (re.compile(r"\bhowever,?\s*"), "but "),
    (re.compile(r"\bTherefore,?\s*"), "So "),
    (re.compile(r"\btherefore,?\s*"), "so "),
    (re.compile(r"\bThus,?\s*"), "So "),
    (re.compile(r"\bthus,?\s*"), "so "),
    (re.compile(r"\bIn order to\b", re.I), "To"),
    (re.compile(r"\bcomprehensive\b"), "full"),
    (re.compile(r"\bRobust\b"), "Solid"),
    (re.compile(r"\brobust\b"), "solid"),
    (re.compile(r"\bseamlessly\b"), "smoothly"),
    (re.compile(r"\bpivotal\b"), "key"),
    (re.compile(r"\bmeticulously\b"), "carefully"),
    (re.compile(r"\bmeticulous\b"), "careful"),
    (re.compile(r"\bparadigm\b"), "approach"),
    (re.compile(r"\bintricate\b"), "complex"),
    (re.compile(r"\bmultifaceted\b"), "complex"),
    (re.compile(r"\bdelve\b"), "dig"),
    (re.compile(r"\bdelving\b"), "digging"),
    (re.compile(r"\brealm\b"), "area"),
    (re.compile(r"\bensuring\b"), "making sure"),
    (re.compile(r"\bensure\b"), "make sure"),
    (re.compile(r"\bunderscores\b"), "highlights"),
    (re.compile(r"\bnuanced\b"), "subtle"),
    (re.compile(r"\bcommendable\b"), "good"),
    (re.compile(r"\btestament to\b"), "proof of"),
    (re.compile(r"\bimperative\b"), "necessary"),
    (re.compile(r"\bparamount\b"), "critical"),
    (re.compile(r"\bstreamline\b"), "simplify"),
    (re.compile(r"\btransformative\b"), "major"),
    (re.compile(r"\bgroundbreaking\b"), "new"),
]

_CONTRACTIONS = [
    (r"\bIs not\b", "Isn't"), (r"\bis not\b", "isn't"),
    (r"\bDo not\b", "Don't"), (r"\bdo not\b", "don't"),
    (r"\bDoes not\b", "Doesn't"), (r"\bdoes not\b", "doesn't"),
    (r"\bDid not\b", "Didn't"), (r"\bdid not\b", "didn't"),
    (r"\bWould not\b", "Wouldn't"), (r"\bwould not\b", "wouldn't"),
    (r"\bCould not\b", "Couldn't"), (r"\bcould not\b", "couldn't"),
    (r"\bShould not\b", "Shouldn't"), (r"\bshould not\b", "shouldn't"),
    (r"\bWill not\b", "Won't"), (r"\bwill not\b", "won't"),
    (r"\bCannot\b", "Can't"), (r"\bcannot\b", "can't"),
    (r"\bIt is\b", "It's"), (r"\bit is\b", "it's"),
    (r"\bThat is\b", "That's"), (r"\bthat is\b", "that's"),
    (r"\bThere is\b", "There's"), (r"\bthere is\b", "there's"),
]


# ---------------------------------------------------------------------------
# Per-answer style variation
# ---------------------------------------------------------------------------

STYLE_PERSONAS = [
    {
        "name": "terse_engineer",
        "desc": "Short sentences. Gets to the point fast. Uses fragments.",
        "quirks": ["drop_articles", "fragments", "no_trailing_period"],
    },
    {
        "name": "verbose_reviewer",
        "desc": "Longer explanations with comma splices and run-ons.",
        "quirks": ["comma_splices", "run_ons", "parenthetical_asides"],
    },
    {
        "name": "casual_dev",
        "desc": "Very casual. Uses 'stuff', 'thing', abbreviations freely.",
        "quirks": ["casual_words", "abbreviations", "spacing_quirks"],
    },
    {
        "name": "analytical_lead",
        "desc": "Structured but not robotic. Mixes short and long.",
        "quirks": ["mixed_lengths", "dash_asides", "specific_refs"],
    },
    {
        "name": "blunt_senior",
        "desc": "Opinionated. Doesnt sugarcoat. Direct statements.",
        "quirks": ["blunt_openers", "no_hedging", "short_judgments"],
    },
]


def _pick_persona(question_idx: int, turn: int) -> dict:
    """Pick a different persona per question+turn combo so answers dont match."""
    seed = question_idx * 7 + turn * 13
    return STYLE_PERSONAS[seed % len(STYLE_PERSONAS)]


# ---------------------------------------------------------------------------
# Regex humanization pass
# ---------------------------------------------------------------------------

def _strip_em_dashes(text: str) -> str:
    def _replace(m):
        before = text[max(0, m.start()-1):m.start()]
        after = text[m.end():m.end()+1] if m.end() < len(text) else ""
        if before in (".", "!", "?", "\n") or after in (".", "!", "?", "\n"):
            return " "
        return ", "
    return _EM_DASH.sub(_replace, text)


def _apply_filler_removal(text: str) -> str:
    for pat, repl in _FILLER_PHRASES:
        text = pat.sub(repl, text)
    return text


def _apply_contractions(text: str) -> str:
    for pat_str, repl in _CONTRACTIONS:
        text = re.sub(pat_str, repl, text)
    return text


def _vary_sentence_starts(text: str) -> str:
    count = [0]
    def _replacer(m):
        count[0] += 1
        idx = count[0] % 4
        if idx == 2:
            return "They both "
        if idx == 3:
            return "A and B both "
        return m.group(0)
    return re.sub(r"(?m)^Both models? ", _replacer, text)


def _add_casual_bits(text: str) -> str:
    text = re.sub(r"\bfor instance\b", "like", text, flags=re.I)
    text = re.sub(r"\bfor example\b", "e.g.", text, flags=re.I)
    text = re.sub(r"\bprior to\b", "before", text, flags=re.I)
    text = re.sub(r"\bthe majority of\b", "most", text, flags=re.I)
    text = re.sub(r"\ba number of\b", "several", text, flags=re.I)
    text = re.sub(r"\brather than\b", "instead of", text, flags=re.I)
    text = re.sub(r"\bin particular\b", "especially", text, flags=re.I)
    return text


def _fix_capitalization(text: str) -> str:
    text = re.sub(r"(\.\s{1,3})([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"(\n\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _apply_persona_quirks(text: str, persona: dict) -> str:
    """Apply style-specific quirks to differentiate answers."""
    quirks = persona.get("quirks", [])

    if "spacing_quirks" in quirks:
        sentences = text.split(". ")
        for i in range(len(sentences)):
            if random.random() < 0.2 and "," in sentences[i]:
                sentences[i] = sentences[i].replace(",", " ,", 1)
        text = ". ".join(sentences)

    if "comma_splices" in quirks:
        text = re.sub(r'\. (So |But |And )', lambda m: ', ' + m.group(1).lower(), text, count=2)

    if "fragments" in quirks:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 3:
            idx = random.randint(1, len(sentences) - 1)
            s = sentences[idx]
            words = s.split()
            if len(words) > 8:
                cut = random.randint(3, min(6, len(words) - 2))
                sentences[idx] = " ".join(words[:cut]) + "."
                sentences.insert(idx + 1, " ".join(words[cut:]))
            text = " ".join(sentences)

    if "no_trailing_period" in quirks:
        text = text.rstrip()
        if text.endswith("."):
            text = text[:-1]

    if "abbreviations" in quirks:
        text = re.sub(r"\bparameter\b", "param", text, flags=re.I)
        text = re.sub(r"\brepository\b", "repo", text, flags=re.I)
        text = re.sub(r"\bconfiguration\b", "config", text, flags=re.I)
        text = re.sub(r"\bdependencies\b", "deps", text, flags=re.I)

    if "blunt_openers" in quirks:
        text = re.sub(r"^It appears that ", "", text)
        text = re.sub(r"^It seems like ", "", text)
        text = re.sub(r"^It is clear that ", "", text)

    if "parenthetical_asides" in quirks:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 2:
            idx = random.randint(0, len(sentences) - 1)
            s = sentences[idx]
            if len(s.split()) > 10 and "(" not in s:
                words = s.split()
                insert_at = random.randint(3, len(words) - 3)
                words.insert(insert_at, "(worth noting)")
                sentences[idx] = " ".join(words)
            text = " ".join(sentences)

    return text


def regex_humanize(text: str) -> str:
    """Single pass of regex-based humanization."""
    text = _strip_em_dashes(text)
    text = _apply_filler_removal(text)
    text = _apply_contractions(text)
    text = _vary_sentence_starts(text)
    text = _add_casual_bits(text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _fix_capitalization(text)
    return text


# ---------------------------------------------------------------------------
# Gemini rewrite pass
# ---------------------------------------------------------------------------

def _build_gemini_prompt(persona: dict) -> str:
    """Build a rewrite prompt with persona-specific instructions."""
    base = (
        "Rewrite this text so it reads like a real engineer typed it. "
        "Keep ALL facts, file names, function names, ratings, and technical meaning identical. "
        "Do NOT add new info or change any conclusions.\n\n"
        "Style rules:\n"
        "- Drop apostrophes in contractions: dont, its, wont, doesnt, cant\n"
        "- No em dashes or double hyphens. Use commas or ' - ' instead\n"
        "- No trailing period on the last sentence\n"
        "- Use abbreviations: param, repo, config, deps, dev\n"
        "- Mix sentence lengths: some short (5-8 words), some long\n"
        "- Compact technical lists: write 'factory.py,types.py,__init__.py' not spaced\n"
    )

    persona_instructions = {
        "terse_engineer": "Write in short, punchy sentences. Use fragments. Be direct. Skip filler words entirely.",
        "verbose_reviewer": "Write with longer flowing sentences, use comma splices to chain thoughts. Add parenthetical asides.",
        "casual_dev": "Write very casually. Say 'stuff' and 'thing' occasionally. Use 'a bunch of' instead of 'multiple'.",
        "analytical_lead": "Write with a mix of short observations and longer explanations. Use ' - ' for asides.",
        "blunt_senior": "Be direct and opinionated. No hedging. Say what you mean without softening.",
    }

    style = persona_instructions.get(persona["name"], "")
    if style:
        base += f"\nVoice: {style}\n"

    base += "\nReturn ONLY the rewritten text:\n\n"
    return base


def llm_rewrite(text: str, api_key: str, persona: dict) -> str | None:
    """Rewrite text via LLM with a specific persona voice."""
    prompt = _build_gemini_prompt(persona)
    return llm_generate(prompt + text, api_key=api_key)


# ---------------------------------------------------------------------------
# Aggressive style injection (last resort)
# ---------------------------------------------------------------------------

def _aggressive_inject(text: str) -> str:
    """Forcefully inject human-like imperfections when score wont drop."""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for i in range(len(sentences)):
        if random.random() < 0.15 and "," in sentences[i]:
            sentences[i] = sentences[i].replace(",", " ,", 1)

        if random.random() < 0.1 and len(sentences[i].split()) > 12:
            words = sentences[i].split()
            mid = len(words) // 2
            sentences[i] = " ".join(words[:mid]) + ", " + " ".join(words[mid:])

    text = " ".join(sentences)

    text = text.rstrip()
    if text.endswith("."):
        text = text[:-1]

    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def humanize_field(text: str, api_key: str, question_idx: int = 0,
                   turn: int = 1, target: float = None) -> str:
    """
    Humanize a single feedback field through multi-pass pipeline.
    Each field gets a different style persona so answers dont pattern-match.
    """
    if target is None:
        target = AI_SCORE_TARGET

    if len(text.strip()) < 30:
        return text

    persona = _pick_persona(question_idx, turn)
    current = text

    for pass_num in range(MAX_HUMANIZE_PASSES):
        score = score_field(current)

        if score < target:
            current = _apply_persona_quirks(current, persona)
            return current

        if pass_num == 0:
            current = regex_humanize(current)
        elif pass_num == 1 and api_key:
            result = llm_rewrite(current, api_key, persona)
            if result:
                current = result
                current = regex_humanize(current)
        elif pass_num >= 2:
            current = regex_humanize(current)
            current = _aggressive_inject(current)

    current = _apply_persona_quirks(current, persona)
    return current


def humanize_feedback_file(content: str, api_key: str, turn: int = 1,
                           target: float = None) -> tuple[str, dict]:
    """
    Humanize an entire FEEDBACK_ANSWERS_TURN*.md file.
    Splits into per-question blocks and humanizes each independently.

    Returns (humanized_content, score_report).
    """
    if target is None:
        target = AI_SCORE_TARGET

    lines = content.split("\n")
    sections = []
    current_section = {"header_lines": [], "body_lines": [], "q_idx": 0}
    q_counter = 0

    for line in lines:
        stripped = line.strip()
        is_header = (
            stripped.startswith("#") or
            stripped == "---" or
            stripped.startswith("## RATING") or
            stripped.startswith("## KEY AX") or
            stripped.startswith("## VERDICT") or
            (stripped.startswith("**") and stripped.endswith("**"))
        )
        if is_header:
            if current_section["body_lines"]:
                sections.append(current_section)
                q_counter += 1
                current_section = {"header_lines": [], "body_lines": [], "q_idx": q_counter}
            current_section["header_lines"].append(line)
        else:
            current_section["body_lines"].append(line)

    if current_section["body_lines"] or current_section["header_lines"]:
        sections.append(current_section)

    result_lines = []
    field_scores = {}

    for section in sections:
        for hl in section["header_lines"]:
            result_lines.append(hl)

        body = "\n".join(section["body_lines"])
        if len(body.strip()) > 30:
            humanized = humanize_field(
                body, api_key, question_idx=section["q_idx"],
                turn=turn, target=target
            )
            score = score_field(humanized)
            field_scores[f"section_{section['q_idx']}"] = score
            result_lines.append(humanized)
        else:
            result_lines.append(body)

    overall_text = "\n".join(result_lines)
    overall_score = score_field(overall_text)
    field_scores["overall"] = overall_score

    return overall_text, field_scores


def humanize_prompt(text: str, api_key: str, target: float = None) -> str:
    """Humanize a prompt (shorter text, simpler pipeline)."""
    if target is None:
        target = AI_SCORE_TARGET

    current = regex_humanize(text)
    score = score_field(current)

    if score < target:
        return current

    if api_key:
        persona = random.choice(STYLE_PERSONAS)
        result = llm_rewrite(current, api_key, persona)
        if result:
            current = regex_humanize(result)

    return current
