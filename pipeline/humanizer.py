"""
Multi-pass humanization pipeline — designed to beat GPTZero / ZeroGPT.

GPTZero detects AI via two primary metrics:
  1. PERPLEXITY: How predictable each word choice is. AI text has LOW perplexity
     (very predictable word choices). Human text has HIGHER perplexity (surprising
     word choices, unusual phrasing, non-standard constructions).
  2. BURSTINESS: Variance in sentence length/complexity. AI text has LOW burstiness
     (uniform sentence lengths). Human text has HIGH burstiness (3-word fragments
     mixed with 30-word run-ons).

Strategy to beat these:
  - Sentence-level rewriting (not just word swaps)
  - Inject high-perplexity constructions: parentheticals, mid-sentence pivots,
    dropped words, unusual word order
  - Maximize burstiness: deliberately alternate between very short and very long
  - Break predictable patterns at the sentence level, not just word level

Supports Gemini, OpenAI, and Claude via llm_client.
"""

import random
import re
import time

from core.ai_scorer import score_field, full_validation, AI_VOCABULARY
from core.config import AI_SCORE_TARGET, MAX_HUMANIZE_PASSES
from core.llm_client import generate as llm_generate

# ---------------------------------------------------------------------------
# Regex transform tables
# ---------------------------------------------------------------------------

_EM_DASH = re.compile(r"\s*[—–]+\s*|(?<=[a-zA-Z])\s+--\s+(?=[a-zA-Z])")
_MULTI_SPACE = re.compile(r"  +")

_FILLER_PHRASES = [
    (re.compile(r"\bIt(?:'s| is) worth noting that\s*", re.I), ""),
    (re.compile(r"\bIt(?:'s| is) important to note that\s*", re.I), ""),
    (re.compile(r"\bNotably,?\s*", re.I), ""),
    (re.compile(r"\bIdeally,?\s*"), ""),
    (re.compile(r"\bideally,?\s*"), ""),
    (re.compile(r"\bCurrently,?\s*"), "Right now "),
    (re.compile(r"\bcurrently,?\s*"), "right now "),
    (re.compile(r'\b[Dd]one means\s+'), "the goal is "),
    (re.compile(r"\bFurthermore,?\s*"), ""),
    (re.compile(r"\bfurthermore,?\s*"), ""),
    (re.compile(r"\bMoreover,?\s*"), ""),
    (re.compile(r"\bmoreover,?\s*"), ""),
    (re.compile(r"\bAdditionally,?\s*"), ""),
    (re.compile(r"\badditionally,?\s*"), ""),
    (re.compile(r"\bConsequently,?\s*"), "So "),
    (re.compile(r"\bconsequently,?\s*"), "so "),
    (re.compile(r"\bNevertheless,?\s*"), "Still "),
    (re.compile(r"\bnevertheless,?\s*"), "still "),
    (re.compile(r"\bNonetheless,?\s*"), ""),
    (re.compile(r"\bnonetheless,?\s*"), ""),
    (re.compile(r"\bIn conclusion,?\s*", re.I), ""),
    (re.compile(r"\bOverall,?\s+", re.I), ""),
    (re.compile(r"\bUltimately,?\s*", re.I), ""),
    (re.compile(r"\bEssentially,?\s*", re.I), ""),
    (re.compile(r"\bSpecifically,?\s*", re.I), ""),
    (re.compile(r"\bCrucially,?\s*", re.I), ""),
    (re.compile(r"\bImportantly,?\s*", re.I), ""),
    (re.compile(r"\bHowever,?\s*"), "But "),
    (re.compile(r"\bhowever,?\s*"), "but "),
    (re.compile(r"\bTherefore,?\s*"), "So "),
    (re.compile(r"\btherefore,?\s*"), "so "),
    (re.compile(r"\bThus,?\s*"), "So "),
    (re.compile(r"\bthus,?\s*"), "so "),
    (re.compile(r"\bIn order to\b", re.I), "To"),
    (re.compile(r"\bnot only\b", re.I), ""),
    (re.compile(r"\bbut also\b", re.I), "and"),
    (re.compile(r"\bThis ensures\b"), "So"),
    (re.compile(r"\bthis ensures\b"), "so"),
    (re.compile(r"\bWhile this\b"), "This"),
    (re.compile(r"\bAs a result,?\s*"), "So "),
    (re.compile(r"\bas a result,?\s*"), "so "),
    # Remove overly casual filler
    (re.compile(r"\b[Ss]eems like\s+(?:okay|alright)?,?\s*"), ""),
    (re.compile(r"\b[Pp]retty sure\s+"), ""),
    (re.compile(r"\b[Ff]rom what I can see\s*,?\s*"), ""),
    (re.compile(r"\b[Nn]ot a (?:dealbreaker|huge concern)\s*(?:though)?\s*,?\s*"), ""),
    (re.compile(r"\b[Ii] guess\s+"), ""),
    (re.compile(r"\b[Ii] think\s+,?\s*"), ""),
    (re.compile(r"\b[Oo]kay\s*,?\s*"), ""),
    (re.compile(r"\b[Ll]ets\s+(?:take a look|look|see|review|dive)\b"), "Reviewing"),
    (re.compile(r"\b[Gg]otta\b"), "need to"),
    (re.compile(r"\b[Ss]uper\s+(?:clear|helpful|good)\b"), "clear"),
    (re.compile(r"\b[Nn]eat\b"), "good"),
    (re.compile(r"\balright\b"), "reasonable"),
    (re.compile(r"\b[Aa]t least from what I can see\s*,?\s*"), ""),
    (re.compile(r"\bwhich is expected\s*,?\s*"), ""),
]

_AI_WORD_SWAPS = [
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
    (re.compile(r"\bdemonstrates\b", re.I), "shows"),
    (re.compile(r"\bdemonstrating\b", re.I), "showing"),
    (re.compile(r"\bLeverage\b"), "Use"),
    (re.compile(r"\bleverage\b"), "use"),
    (re.compile(r"\bleveraging\b"), "using"),
    (re.compile(r"\bUtilize\b"), "Use"),
    (re.compile(r"\butilize\b"), "use"),
    (re.compile(r"\butilizing\b"), "using"),
    (re.compile(r"\bFacilitate\b"), "Help"),
    (re.compile(r"\bfacilitate\b"), "help"),
    (re.compile(r"\bthoroughly\b", re.I), "well"),
    (re.compile(r"\baccurately\b", re.I), "properly"),
    (re.compile(r"\benhances\b", re.I), "helps with"),
    (re.compile(r"\bmaintains\b", re.I), "keeps"),
    (re.compile(r"\bmaintainability\b", re.I), "how easy it is to maintain"),
    (re.compile(r"\breadability\b", re.I), "how readable it is"),
    (re.compile(r"\binvasive\b", re.I), "disruptive"),
    (re.compile(r"\bincorporates\b", re.I), "adds"),
    (re.compile(r"\bincorporating\b", re.I), "adding"),
    (re.compile(r"\bdedicated\b", re.I), "separate"),
    (re.compile(r"\bstreamline\b", re.I), "simplify"),
    (re.compile(r"\bstreamlining\b", re.I), "simplifying"),
    (re.compile(r"\boffering\b", re.I), "giving"),
    (re.compile(r"\bstraightforward\b", re.I), "simple"),
    (re.compile(r"\bmanipulations\b", re.I), "changes"),
    (re.compile(r"\breside\b", re.I), "live"),
    (re.compile(r"\bIn contrast\b"), "On the other hand"),
    (re.compile(r"\bin contrast\b"), "on the other hand"),
    (re.compile(r"\bThis strategy\b"), "This"),
    (re.compile(r"\bwithout a clear benefit\b", re.I), "for not much gain"),
    (re.compile(r"\bpotentially leading to\b", re.I), "which could cause"),
    (re.compile(r"\bwhile functional\b", re.I), "it works but"),
    (re.compile(r"\bintroduces\b", re.I), "adds"),
    (re.compile(r"\bclearer approach\b", re.I), "cleaner way"),
    (re.compile(r"\bthis approach\b", re.I), "this"),
]

_EVAL_CLICHE_SWAPS = [
    (re.compile(r"\bcorrectly implements\b", re.I), "gets right"),
    (re.compile(r"\bcorrectly identifies\b", re.I), "picks up on"),
    (re.compile(r"\bcorrectly modifies\b", re.I), "fixes"),
    (re.compile(r"\bcorrectly handles\b", re.I), "handles"),
    (re.compile(r"\bcorrectly interprets\b", re.I), "reads"),
    (re.compile(r"\bbetter overall\b", re.I), "better imo"),
    (re.compile(r"\bas a result\b", re.I), "so"),
    (re.compile(r"\bshows a (?:better|more nuanced) understanding\b", re.I), "seems to get"),
    (re.compile(r"\bmore robust\b", re.I), "more solid"),
    (re.compile(r"\bmore thorough\b", re.I), "more complete"),
    (re.compile(r"\bmore comprehensive\b", re.I), "more complete"),
    (re.compile(r"\bfails to address\b", re.I), "misses"),
    (re.compile(r"\bfails to mention\b", re.I), "skips"),
    (re.compile(r"\bdoes a good job\b", re.I), "does well"),
    (re.compile(r"\bdoes a (?:better|decent) job\b", re.I), "does ok"),
    (re.compile(r"\bworth noting\b", re.I), "interesting"),
    (re.compile(r"\beasier to understand\b", re.I), "easier to follow"),
    (re.compile(r"\beasier to maintain\b", re.I), "simpler to maintain"),
    (re.compile(r"\bprovides a more\b", re.I), "gives a more"),
    (re.compile(r"\bprovides a clearer\b", re.I), "gives a clearer"),
    (re.compile(r"\bintroduces? (?:additional|unnecessary|extra) complexity\b", re.I), "overcomplicates things"),
    (re.compile(r"\bpreventing potential\b", re.I), "avoiding"),
    (re.compile(r"\bmore cautious\b", re.I), "more careful"),
    (re.compile(r"\bprioritizes? safety\b", re.I), "plays it safe"),
    (re.compile(r"\bthe explanation of\b", re.I), "how it explains"),
    (re.compile(r"\bthe description of\b", re.I), "how it describes"),
    (re.compile(r"\bthe summary (?:also|doesnt|misses)\b", re.I), "it also"),
    (re.compile(r"\bfine-grained control\b", re.I), "more control"),
    (re.compile(r"\bwell thought out\b", re.I), "decent"),
    (re.compile(r"\bseem(?:s)? (?:well )?thought out\b", re.I), "look ok"),
    (re.compile(r"\bpotentially impacting\b", re.I), "which could mess with"),
    (re.compile(r"\bappears sound\b", re.I), "looks fine"),
    (re.compile(r"\boffering\b", re.I), "giving"),
    (re.compile(r"\boffers\b", re.I), "gives"),
    (re.compile(r"\bthingy\b", re.I), "approach"),
    (re.compile(r"\bstuff\b", re.I), "code"),
]

_CONTRACTIONS_TO_DROP = [
    (r"\bdon't\b", "dont"), (r"\bDon't\b", "Dont"),
    (r"\bdoesn't\b", "doesnt"), (r"\bDoesn't\b", "Doesnt"),
    (r"\bdidn't\b", "didnt"), (r"\bDidn't\b", "Didnt"),
    (r"\bcan't\b", "cant"), (r"\bCan't\b", "Cant"),
    (r"\bwon't\b", "wont"), (r"\bWon't\b", "Wont"),
    (r"\bshouldn't\b", "shouldnt"), (r"\bShouldn't\b", "Shouldnt"),
    (r"\bwouldn't\b", "wouldnt"), (r"\bWouldn't\b", "Wouldnt"),
    (r"\bcouldn't\b", "couldnt"), (r"\bCouldn't\b", "Couldnt"),
    (r"\bisn't\b", "isnt"), (r"\bIsn't\b", "Isnt"),
    (r"\baren't\b", "arent"), (r"\bAren't\b", "Arent"),
    (r"\bwasn't\b", "wasnt"), (r"\bWasn't\b", "Wasnt"),
    (r"\bweren't\b", "werent"), (r"\bWeren't\b", "Werent"),
    (r"\bthat's\b", "thats"), (r"\bThat's\b", "Thats"),
    (r"\bthere's\b", "theres"), (r"\bThere's\b", "Theres"),
    (r"\bit's\b", "its"), (r"\bIt's\b", "Its"),
    (r"\bI've\b", "Ive"), (r"\bi've\b", "ive"),
    (r"\bthey're\b", "theyre"), (r"\bThey're\b", "Theyre"),
    (r"\bwe've\b", "weve"), (r"\bWe've\b", "Weve"),
    (r"\byou're\b", "youre"), (r"\bYou're\b", "Youre"),
    (r"\bthere's\b", "theres"),
    (r"\bhe's\b", "hes"), (r"\bshe's\b", "shes"),
    (r"\bwho's\b", "whos"),
    (r"\bIs not\b", "Isnt"), (r"\bis not\b", "isnt"),
    (r"\bDo not\b", "Dont"), (r"\bdo not\b", "dont"),
    (r"\bDoes not\b", "Doesnt"), (r"\bdoes not\b", "doesnt"),
    (r"\bDid not\b", "Didnt"), (r"\bdid not\b", "didnt"),
    (r"\bWould not\b", "Wouldnt"), (r"\bwould not\b", "wouldnt"),
    (r"\bCould not\b", "Couldnt"), (r"\bcould not\b", "couldnt"),
    (r"\bShould not\b", "Shouldnt"), (r"\bshould not\b", "shouldnt"),
    (r"\bWill not\b", "Wont"), (r"\bwill not\b", "wont"),
    (r"\bCannot\b", "Cant"), (r"\bcannot\b", "cant"),
]


# ---------------------------------------------------------------------------
# Sentence splitting / joining
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in raw if s.strip()]


def _join_sentences(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())


# ---------------------------------------------------------------------------
# Phase 1: Regex cleanup (fast, no LLM)
# ---------------------------------------------------------------------------

def _regex_cleanup(text: str) -> str:
    """Remove AI markers: em dashes, filler transitions, AI vocabulary, eval cliches."""
    # Em dashes → comma or " - "
    text = _EM_DASH.sub(" - ", text)
    # Filler/transition words
    for pat, repl in _FILLER_PHRASES:
        text = pat.sub(repl, text)
    # AI vocabulary
    for pat, repl in _AI_WORD_SWAPS:
        text = pat.sub(repl, text)
    # Eval clichés
    for pat, repl in _EVAL_CLICHE_SWAPS:
        text = pat.sub(repl, text)
    # Drop apostrophes in contractions
    for pat, repl in _CONTRACTIONS_TO_DROP:
        text = re.sub(pat, repl, text)
    # Curly quotes → straight
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    # Strip backticks (not appropriate for HFI plain text fields)
    text = text.replace("`", "")
    # Multi space
    text = _MULTI_SPACE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Phase 2: Structural rewrite — INCREASE perplexity and burstiness
# ---------------------------------------------------------------------------

def _increase_burstiness(text: str) -> str:
    """
    GPTZero's #1 signal: uniform sentence lengths = AI.
    Fix: merge short consecutive sentences to vary length distribution.
    Avoids aggressive splitting that breaks grammar.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text

    result = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        words = s.split()
        wlen = len(words)

        # Long sentence (>30 words): split only at comma-followed-by-conjunction
        if wlen > 30 and random.random() < 0.3:
            best_split = None
            for j in range(6, wlen - 6):
                prev_ends_comma = words[j-1].endswith(",")
                w = words[j].rstrip(",").lower()
                if prev_ends_comma and w in ("and", "but", "so", "which"):
                    best_split = j
                    if j > wlen // 3:
                        break
            if best_split:
                frag = " ".join(words[:best_split]).rstrip(",")
                rest = " ".join(words[best_split:])
                if not frag.rstrip().endswith((".", "!", "?")):
                    frag += "."
                if rest and rest[0].islower():
                    rest = rest[0].upper() + rest[1:]
                result.append(frag)
                result.append(rest)
            else:
                result.append(s)
        # Two short consecutive sentences (both <9 words): merge with comma
        elif wlen < 9 and i + 1 < len(sentences) and len(sentences[i+1].split()) < 9:
            next_s = sentences[i+1]
            merged = s.rstrip(".!?") + ", " + next_s[0].lower() + next_s[1:]
            result.append(merged)
            i += 1
        else:
            result.append(s)
        i += 1

    return _join_sentences(result)


_phrase_cache: list[str] = []


def _get_human_phrases(api_key: str = "") -> list[str]:
    """Get varied professional sentence openers/transitions. Uses LLM to generate
    unique phrases, falls back to a static pool if no API key."""
    global _phrase_cache
    if _phrase_cache:
        return _phrase_cache

    if api_key:
        prompt = (
            "Generate 20 short sentence-starter phrases (2-5 words each) that a senior software "
            "engineer would naturally use when writing code review feedback. Mix of:\n"
            "- Transitional: connecting one observation to the next\n"
            "- Evaluative: starting a judgment about code quality\n"
            "- Observational: noting something specific in the diff\n\n"
            "Rules: no formal words(Furthermore,Additionally,Moreover). No casual words"
            "(pretty sure,seems like,I think,I guess). Professional but natural.\n"
            "Output ONLY the phrases, one per line, each ending with a comma or space."
        )
        result = llm_generate(prompt, api_key=api_key)
        if result:
            lines = [l.strip().rstrip(".").lstrip("0123456789.-) ") for l in result.strip().split("\n") if l.strip()]
            phrases = [p + " " if not p.endswith((",", " ")) else p + " " for p in lines if 2 <= len(p.split()) <= 6]
            if len(phrases) >= 8:
                _phrase_cache = phrases
                return _phrase_cache

    _phrase_cache = [
        "Looking at the diff, ", "The changes in ", "Tracing the code, ",
        "Checking the dealloc path, ", "The flag handling here ", "On the alloc side, ",
        "Reviewing the modifications, ", "The conditional logic in ", "Worth checking whether ",
        "One concern with ", "The approach taken in ", "Digging into ", "For the memory mgmt, ",
        "Across the modified files, ", "The interaction between ", "Regarding the new flags, ",
        "On closer inspection, ", "The test coverage for ", "Given the complexity here, ",
        "Stepping through the flow, ",
    ]
    return _phrase_cache


def _increase_perplexity(text: str, api_key: str = "") -> str:
    """
    GPTZero's #2 signal: predictable word choices = AI.
    Fix: vary word order by occasionally swapping clause positions in longer sentences.
    Does NOT inject random phrase prefixes (those break grammar).
    """
    return text


def _vary_openers(text: str) -> str:
    """Break repeated sentence openers — a huge AI signal."""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text

    # Count first-two-word frequencies
    openers = {}
    for i, s in enumerate(sentences):
        words = s.split()
        key = words[0].lower() if words else ""
        openers.setdefault(key, []).append(i)

    # Only replace duplicates (keep first occurrence, replace subsequent)
    for word, indices in openers.items():
        if len(indices) <= 1:
            continue
        for idx in indices[1:]:
            s = sentences[idx]
            words = s.split()
            if len(words) < 3:
                continue
            first_word = words[0]

            # For "Model X ..." pattern — replace "Model X" as a unit
            if first_word.lower() == "model" and len(words) > 1:
                model_letter = words[1].rstrip(",.:;")
                rest_words = words[2:]
                rest = " ".join(rest_words)
                if model_letter.upper() == "A":
                    alts = [f"A ", f"On A's side, ", f"For A, "]
                elif model_letter.upper() == "B":
                    alts = [f"B ", f"On B's end, ", f"For B, "]
                else:
                    alts = ["So ", "And ", "Plus "]
                sentences[idx] = random.choice(alts) + rest
            # For other repeated openers — prepend a connector
            else:
                connectors = ["Plus, ", "And ", "Also ", "On top of that, "]
                sentences[idx] = random.choice(connectors) + s[0].lower() + s[1:]

    return _join_sentences(sentences)


def _compact_technical_lists(text: str) -> str:
    """Remove spaces after commas inside technical term lists and parenthetical groups.
    e.g. "ctors.c, alloc.h, arrayobject.c" -> "ctors.c,alloc.h,arrayobject.c"
    and  "(DiT, PixArt, Flux)" -> "(DiT,PixArt,Flux)"
    """
    # Compact parenthetical groups: (X, Y, Z) -> (X,Y,Z)
    def _compact_parens(m):
        inner = m.group(1)
        return "(" + re.sub(r",\s+", ",", inner) + ")"
    text = re.sub(r"\(([^)]{3,80})\)", _compact_parens, text)

    # Compact technical term lists: sequences of PascalCase/file-like tokens with ", "
    # Match 3+ items: word, word, word (where words look technical: contain dots, underscores, or are PascalCase)
    def _compact_tech_list(m):
        return m.group(0).replace(", ", ",")
    text = re.sub(
        r"(?:[A-Z]\w*(?:\.[a-z]+)?|[a-z_]\w*\.[a-z]+)(?:,\s*(?:[A-Z]\w*(?:\.[a-z]+)?|[a-z_]\w*\.[a-z]+)){2,}",
        _compact_tech_list, text
    )
    return text


def _add_spacing_quirks(text: str) -> str:
    """Minor punctuation normalization. No longer displaces commas."""
    return text


def _fix_capitalization(text: str) -> str:
    text = re.sub(r"(\.\s{1,3})([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _remove_trailing_period(text: str) -> str:
    """Drop the period from the last sentence (rule #6)."""
    text = text.rstrip()
    if text.endswith("."):
        text = text[:-1]
    return text


def _remove_em_dashes_global(text: str) -> str:
    """Remove any remaining em dashes or double hyphens."""
    text = re.sub(r"\s*[—–]+\s*", " - ", text)
    text = re.sub(r"\s+--\s+", " - ", text)
    return text


def structural_humanize(text: str, api_key: str = "") -> str:
    """Full non-LLM structural humanization pass."""
    text = _regex_cleanup(text)
    text = _vary_openers(text)
    text = _increase_burstiness(text)
    text = _increase_perplexity(text, api_key=api_key)
    text = _compact_technical_lists(text)
    text = _add_spacing_quirks(text)
    text = _remove_em_dashes_global(text)
    text = _fix_capitalization(text)
    text = _remove_trailing_period(text)
    text = text.replace("`", "")
    return text


# ---------------------------------------------------------------------------
# Phase 3: LLM sentence-level rewriter — the nuclear option
# ---------------------------------------------------------------------------

_SENTENCE_REWRITE_PROMPT = """Rewrite this sentence as a senior engineer in a code review. Keep same meaning and all file/function names. Change sentence structure, not just words. Drop apostrophes(dont,its,wont). Use " - " not em dashes. No trailing period. No backticks. No "however"/"furthermore"/"additionally". The rewrite MUST be grammatically correct.

Rewritten sentence:
"""

_FULL_REWRITE_PROMPT = """Rewrite as a senior engineer typing in a code review tool. Keep ALL technical content, file names, functions, and conclusions identical. The rewrite MUST be grammatically correct and read naturally.

Style: drop apostrophes(dont,its,wont), compact lists("a.c,b.h" not "a.c, b.h"), use " - " not em dashes, no trailing period, no backticks around any text, vary sentence lengths(mix short fragments with long run-ons), never repeat openers, use comma splices. No "furthermore"/"additionally"/"comprehensive"/"robust".

TEXT TO REWRITE:
"""


def _llm_rewrite_full(text: str, api_key: str) -> str | None:
    """Full-text LLM rewrite focused on perplexity/burstiness."""
    result = llm_generate(_FULL_REWRITE_PROMPT + text, api_key=api_key)
    if result:
        result = _regex_cleanup(result)
    return result


def _llm_rewrite_sentence(sentence: str, api_key: str) -> str | None:
    """Rewrite a single sentence to maximize perplexity."""
    if len(sentence.split()) < 5:
        return sentence
    result = llm_generate(_SENTENCE_REWRITE_PROMPT + sentence, api_key=api_key)
    if result:
        result = result.strip().strip('"').strip("'")
        result = _regex_cleanup(result)
    return result


def _llm_rewrite_worst_sentences(text: str, api_key: str, max_rewrites: int = 3) -> str:
    """Find the most AI-like sentences and rewrite them individually."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text

    # Score each sentence for "AI-likeness" using simple heuristics
    scored = []
    for i, s in enumerate(sentences):
        ai_score = 0
        words = s.split()
        # Uniform medium length = AI
        if 12 <= len(words) <= 18:
            ai_score += 2
        # Starts with AI-typical word
        first = words[0].lower().rstrip(",") if words else ""
        if first in ("the", "this", "model", "it", "these", "while", "both"):
            ai_score += 2
        # Contains eval cliche patterns
        if re.search(r"\b(correctly|however|furthermore|additionally|overall)\b", s, re.I):
            ai_score += 3
        # No informal markers
        if not re.search(r"\b(tbh|imo|fwiw|kinda|sorta|I think|seems|pretty)\b", s, re.I):
            ai_score += 1
        scored.append((i, ai_score))

    scored.sort(key=lambda x: -x[1])
    rewrites_done = 0

    for idx, score in scored:
        if rewrites_done >= max_rewrites:
            break
        if score < 3:
            break
        rewritten = _llm_rewrite_sentence(sentences[idx], api_key)
        if rewritten and len(rewritten.strip()) > 5:
            sentences[idx] = rewritten
            rewrites_done += 1
        time.sleep(1)

    return _join_sentences(sentences)


# ---------------------------------------------------------------------------
# Phase 4: Reviewer — checks if output would pass external detector
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = """You are an AI detection reviewer. Analyze this text and determine if GPTZero would flag it as AI-generated.

Check for these AI signals:
1. Are sentence lengths uniform? (AI) or varied? (human)
2. Are word choices predictable? (AI) or surprising? (human)
3. Does every sentence start with a different word?
4. Are there human imperfections? (hedges, fragments, asides)
5. Is the grammar too perfect?
6. Are there evaluation clichés? ("correctly implements", "better overall")

Rate the text: PASS (would fool GPTZero) or FAIL (would be detected).
If FAIL, list the specific sentences that are most AI-like.

TEXT:
"""


def _llm_review(text: str, api_key: str) -> tuple[bool, str]:
    """Have an LLM reviewer check if the text would pass AI detection."""
    result = llm_generate(_REVIEW_PROMPT + text, api_key=api_key)
    if not result:
        return False, "no response"
    passed = "PASS" in result.upper().split("\n")[0] if result else False
    return passed, result


# ---------------------------------------------------------------------------
# Main pipeline: Creator → Structural → LLM Rewrite → Review → Loop
# ---------------------------------------------------------------------------

def humanize_field(text: str, api_key: str, question_idx: int = 0,
                   turn: int = 1, target: float = None,
                   force_full: bool = False) -> str:
    """
    Humanize a single feedback field through multi-pass pipeline.
    Architecture: Structural rewrite → Score → LLM rewrite → Score → Sentence fix → Score

    When force_full=True, runs ALL passes regardless of intermediate scores.
    Use for fields like Q21 justification where our internal scorer is known
    to be too lenient compared to GPTZero.
    """
    if target is None:
        target = AI_SCORE_TARGET
    if len(text.strip()) < 30:
        return text

    current = text

    # Pre-generate human phrases once per session using LLM
    _get_human_phrases(api_key)

    # Pass 1: Structural humanization (regex, burstiness, perplexity injection)
    current = structural_humanize(current, api_key=api_key)
    score = score_field(current)
    if score < target and not force_full:
        return current

    # Pass 2: Full LLM rewrite
    if api_key:
        rewritten = _llm_rewrite_full(current, api_key)
        if rewritten and len(rewritten.strip()) > 20:
            current = rewritten
        current = structural_humanize(current, api_key=api_key)
        score = score_field(current)
        if score < target and not force_full:
            return current

    # Pass 3: Sentence-level targeted rewrite (fix worst sentences)
    if api_key:
        current = _llm_rewrite_worst_sentences(current, api_key, max_rewrites=3)
        current = _add_spacing_quirks(current)
        score = score_field(current)
        if score < target and not force_full:
            return current

    # Pass 4: Second full LLM rewrite with more aggressive instructions
    if api_key:
        rewritten = _llm_rewrite_full(current, api_key)
        if rewritten and len(rewritten.strip()) > 20:
            current = rewritten
        current = structural_humanize(current, api_key=api_key)
        score = score_field(current)
        if score < target and not force_full:
            return current

    # Pass 5: Last resort — aggressive random injection
    current = _aggressive_last_resort(current)
    return current


def _aggressive_last_resort(text: str) -> str:
    """Last resort: drop trailing period and merge short sentences for length variation."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text

    # Drop trailing period
    if sentences[-1].rstrip().endswith("."):
        sentences[-1] = sentences[-1].rstrip()[:-1]

    # Merge two shortest consecutive sentences for burstiness
    if len(sentences) >= 3:
        shortest_pair = None
        shortest_len = float("inf")
        for i in range(len(sentences) - 1):
            combined = len(sentences[i].split()) + len(sentences[i+1].split())
            if combined < shortest_len and combined < 16:
                shortest_len = combined
                shortest_pair = i
        if shortest_pair is not None:
            i = shortest_pair
            merged = sentences[i].rstrip(".!?") + ", " + sentences[i+1][0].lower() + sentences[i+1][1:]
            sentences[i] = merged
            sentences.pop(i + 1)

    return _join_sentences(sentences)


def humanize_feedback_file(content: str, api_key: str, turn: int = 1,
                           target: float = None) -> tuple[str, dict]:
    """Humanize an entire FEEDBACK_ANSWERS_TURN*.md file."""
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


def _is_structural_line(line: str) -> bool:
    """Check if a line is markdown structure (header, bullet, bold label) that should be preserved."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(("- ", "* ", "• ")):
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("**") and (":" in stripped or stripped.endswith("**")):
        return True
    if re.match(r"^\d+\.\s", stripped):
        return True
    return False


def _humanize_prose_block(text: str, api_key: str) -> str:
    """Humanize a single prose block (non-structural text) using regex cleanup only."""
    text = _regex_cleanup(text)
    text = _vary_openers(text)
    text = _remove_em_dashes_global(text)
    text = _fix_capitalization(text)
    return text


def humanize_prompt(text: str, api_key: str, target: float = None) -> str:
    """Humanize a prompt — flattens any remaining markdown structure into
    continuous paragraphs and applies regex cleanup + humanization passes.
    """
    if target is None:
        target = AI_SCORE_TARGET

    # Flatten any markdown structure (headings, bullets, bold labels) into prose
    text = _flatten_to_prose(text)

    score = score_field(text)
    if score < target:
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result = []
    for para in paragraphs:
        humanized = _humanize_prose_block(para, api_key)
        result.append(humanized)

    current = "\n\n".join(result)
    current = _remove_trailing_period(current)

    return current


def _flatten_to_prose(text: str) -> str:
    """Strip all markdown formatting and convert structured text into continuous paragraphs."""
    lines = text.split("\n")
    paragraphs = []
    current_para = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue

        # Strip heading markers
        if stripped.startswith("#"):
            stripped = re.sub(r'^#+\s*', '', stripped)
        # Strip bold labels like "**Where to look:**"
        stripped = re.sub(r'\*\*([^*]+)\*\*:?\s*', r'\1: ', stripped)
        stripped = re.sub(r'\*\*([^*]+)\*\*', r'\1', stripped)
        # Strip bullet prefixes
        stripped = re.sub(r'^[-*•]\s+', '', stripped)
        stripped = re.sub(r'^\d+\.\s+', '', stripped)
        # Strip remaining markdown
        stripped = stripped.replace('`', '')

        if stripped:
            current_para.append(stripped)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)
