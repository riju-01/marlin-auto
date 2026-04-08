"""
AI detection scoring engine — calibrated against GPTZero / ZeroGPT / Originality.

External AI detectors look for:
  - Predictable sentence structure & uniform length
  - Perfect grammar with zero typos or self-corrections
  - Formulaic compare-contrast patterns ("Model A... Model B...")
  - Evaluation clichés ("correctly implements", "better overall")
  - High type-token ratio (too many unique "fancy" words)
  - No fragments, asides, or incomplete thoughts
  - Repeated opener words across sentences
  - Consistent punctuation (no spacing quirks)
  - Lack of first-person hedging ("I think", "not sure but")

Each sub-score is 0-1 (0 = human-like, 1 = AI-like).
Weighted average gives overall score. Target: keep below 0.30.
"""

import math
import re
from collections import Counter

AI_VOCABULARY = {
    # Classic LLM vocabulary
    "delve", "crucial", "comprehensive", "facilitate", "furthermore",
    "moreover", "additionally", "consequently", "nevertheless", "nonetheless",
    "robust", "seamless", "paradigm", "leverage", "utilize", "intricate",
    "multifaceted", "tapestry", "landscape", "realm", "pivotal", "meticulous",
    "notably", "essentially", "fundamentally", "significantly", "interestingly",
    "demonstrating", "demonstrates", "ensuring", "underscores", "underscoring",
    "encompasses", "nuanced", "commendable", "noteworthy",
    "fostering", "harboring", "elucidating", "streamline", "synergy",
    "holistic", "overarching", "underpinning", "spearheading", "culminating",
    "embark", "intricacies", "aligns", "harnessing", "navigating",
    "orchestrating", "bolstering", "augmenting", "catalyzing", "epitomize",
    "groundbreaking", "transformative", "salient", "discerning",
    "illuminating", "testament", "imperative", "paramount", "indispensable",
    "endeavor", "proficiency", "adept", "aptly", "poised",
    "effectively", "particularly", "specifically", "accordingly",
    "implementation", "functionality", "thoroughly", "accurately",
    # Formal academic words GPTZero flags but classic lists miss
    "enhances", "maintains", "readability", "maintainability",
    "invasive", "incorporates", "introduces", "dedicated",
    "streamlining", "offering", "strategy", "straightforward",
    "manipulations", "reside", "determine",
    "clarity", "refined", "functional", "lacking",
    "potentially", "evident", "respective", "modifications",
    "incorporating", "exhibited", "warranted", "necessitates",
    "pertaining", "aforementioned", "delineated", "constitutes",
}

AI_TRANSITIONS = {
    "however", "therefore", "thus", "moreover", "furthermore",
    "additionally", "consequently", "nevertheless", "nonetheless",
    "specifically", "notably", "essentially", "fundamentally",
    "ultimately", "importantly", "significantly", "interestingly",
    "crucially", "overall",
}

EVAL_CLICHES = [
    r"\bcorrectly\s+(?:implements?|identifies?|modifies?|handles?|addresses?)\b",
    r"\bbetter overall\b",
    r"\bas a result\b",
    r"\bshows a better understanding\b",
    r"\bmore (?:robust|thorough|comprehensive|complete)\b",
    r"\bfails to (?:address|mention|identify|highlight)\b",
    r"\bdoes a (?:good|better|great|solid) job\b",
    r"\bkey (?:optimization|improvement|change|aspect)\b",
    r"\bworth noting\b",
    r"\bmore (?:explicit|concise|verbose|readable)\b",
    r"\bclearer (?:summary|explanation|description)\b",
    r"\beasier to (?:understand|follow|maintain|read)\b",
    r"\bprovides a (?:more|clearer|better)\b",
    r"\bintroduces (?:additional|unnecessary|extra) (?:complexity|logic|overhead)\b",
    r"\bpreventing potential\b",
    r"\bmore (?:cautious|careful|conservative)\b",
    r"\bprioritize[sd]? safety\b",
    # Formal connectors GPTZero flags
    r"\bin contrast\b",
    r"\bthis strategy\b",
    r"\bwithout a clear benefit\b",
    r"\bpotentially leading to\b",
    r"\bwhile functional\b",
    r"\boffering more (?:clarity|control|flexibility)\b",
    r"\benhances? (?:readability|maintainability|clarity|safety)\b",
    r"\bstreamlining the\b",
    r"\badding complexity\b",
    r"\bless invasive\b",
    r"\ba clearer approach\b",
    r"\brefin(?:ed|ing) logic\b",
]

META_PHRASES = [
    r"\bthe (?:explanation|description|summary|analysis) (?:of|is)\b",
    r"\bthe summary (?:also|doesnt|misses)\b",
    r"\bmodel [ab](?:'s| also| does| shows| has| provides| correctly)\b",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", text.lower())


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in raw if len(s.split()) >= 3]


def _score_ai_vocabulary(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in AI_VOCABULARY)
    return min(hits / max(len(tokens), 1) * 50, 1.0)


def _score_sentence_uniformity(sentences: list[str]) -> float:
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean_len
    if cv < 0.2:
        return 1.0
    if cv < 0.3:
        return 0.8
    if cv < 0.4:
        return 0.5
    if cv < 0.5:
        return 0.2
    return 0.0


def _score_transition_density(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    hits = 0
    for s in sentences:
        first_word = s.strip().split()[0].lower().rstrip(",") if s.strip() else ""
        if first_word in AI_TRANSITIONS:
            hits += 1
    return min(hits / len(sentences) * 3, 1.0)


def _score_contraction_absence(text: str) -> float:
    standard = len(re.findall(r"\b\w+n't\b|\b\w+'(?:ve|re|ll|d|s|m)\b", text))
    dropped = len(re.findall(
        r"\b(?:dont|doesnt|didnt|cant|wont|shouldnt|wouldnt|couldnt|"
        r"isnt|arent|wasnt|werent|hasnt|havent|hadnt|"
        r"thats|theres|its|ive|youre|youve|theyre|weve|"
        r"itd|itll|thatll|whats|heres|whos)\b", text, re.I
    ))
    contractions = standard + dropped
    words = len(text.split())
    if words < 20:
        return 0.0
    ratio = contractions / words
    if ratio > 0.02:
        return 0.0
    if ratio > 0.01:
        return 0.3
    if ratio > 0.005:
        return 0.6
    return 0.9


def _score_em_dash_usage(text: str) -> float:
    em_dashes = len(re.findall(r"[—–]|\s--\s", text))
    sentences = len(_split_sentences(text))
    if sentences == 0:
        return 0.0
    ratio = em_dashes / sentences
    if ratio > 0.3:
        return 1.0
    if ratio > 0.15:
        return 0.6
    if ratio > 0.05:
        return 0.2
    return 0.0


def _score_vocab_diversity(tokens: list[str]) -> float:
    if len(tokens) < 30:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    if ttr > 0.75:
        return 0.8
    if ttr > 0.68:
        return 0.4
    return 0.0


def _score_paragraph_uniformity(text: str) -> float:
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 5]
    if len(paragraphs) < 3:
        return 0.0
    lengths = [len(p.split()) for p in paragraphs]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean_len
    if cv < 0.15:
        return 1.0
    if cv < 0.25:
        return 0.7
    if cv < 0.4:
        return 0.3
    return 0.0


def _score_formality(text: str) -> float:
    contractions = len(re.findall(r"\b\w+n't\b|\b\w+'(?:ve|re|ll|d|s|m)\b", text))
    dropped_contr = len(re.findall(
        r"\b(?:dont|doesnt|didnt|cant|wont|shouldnt|isnt|thats|its|ive)\b", text, re.I
    ))
    informal = len(re.findall(
        r"\b(?:gonna|gotta|kinda|sorta|y'all|dunno|wanna|yeah|nah|ok|okay|"
        r"tbh|imo|fwiw|btw|lol|hmm|idk|ngl|tho|prolly)\b", text, re.I
    ))
    dashes_as_pauses = len(re.findall(r" - ", text))
    parens = len(re.findall(r"\(", text))
    hedges = len(re.findall(
        r"\b(?:I think|not sure|maybe|probably|might be|could be|I guess|"
        r"feels like|seems like|looks like|hard to say)\b", text, re.I
    ))
    words = len(text.split())
    if words < 20:
        return 0.0
    human_signals = (contractions + dropped_contr + informal +
                     dashes_as_pauses + parens + hedges)
    ratio = human_signals / words
    if ratio > 0.04:
        return 0.0
    if ratio > 0.025:
        return 0.1
    if ratio > 0.015:
        return 0.3
    if ratio > 0.005:
        return 0.6
    return 0.95


def _score_opener_repetition(text: str) -> float:
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 0.0
    openers = []
    for s in sentences:
        words = s.strip().split()[:2]
        openers.append(" ".join(w.lower() for w in words))
    counts = Counter(openers)
    if not counts:
        return 0.0
    most_common_count = counts.most_common(1)[0][1]
    ratio = most_common_count / len(openers)
    if ratio > 0.5:
        return 1.0
    if ratio > 0.35:
        return 0.8
    if ratio > 0.25:
        return 0.6
    if ratio > 0.15:
        return 0.3
    return 0.0


_AI_PHRASE_PATTERNS = [
    r"\bCurrently,?\s",
    r"\bIdeally,?\s",
    r"\bDone means\b",
    r"\bThis is especially\b",
    r"\bThis should be done\b",
    r"\bIt is not always clear\b",
    r"\bmaking it harder to\b",
    r"\bwhich adds overhead\b",
    r"\bIn summary,?\s",
    r"\bTo summarize,?\s",
    r"\bIn this context,?\s",
    r"\bIt is worth mentioning\b",
    r"\bFor standard\b.*\bshould be\b",
    r"\bin a way that does not break\b",
    r"\bmore solid and easier\b",
    r"\bleading to potential\b",
    r"\bThis ensures\b",
    r"\bBy doing so\b",
    r"\bAs a result\b",
    r"\bWith this approach\b",
]


def _score_ai_phrasing(text: str) -> float:
    if len(text) < 80:
        return 0.0
    hits = 0
    for pat in _AI_PHRASE_PATTERNS:
        if re.search(pat, text, re.I):
            hits += 1
    if hits >= 5:
        return 1.0
    if hits >= 3:
        return 0.7
    if hits >= 2:
        return 0.5
    if hits >= 1:
        return 0.25
    return 0.0


def _score_eval_cliches(text: str) -> float:
    """Detect evaluation clichés that scream AI-written review."""
    if len(text) < 60:
        return 0.0
    hits = 0
    for pat in EVAL_CLICHES:
        if re.search(pat, text, re.I):
            hits += 1
    if hits >= 5:
        return 1.0
    if hits >= 3:
        return 0.8
    if hits >= 2:
        return 0.6
    if hits >= 1:
        return 0.35
    return 0.0


def _score_meta_phrases(text: str) -> float:
    """Detect meta-commentary phrases like 'The explanation of X is...'"""
    if len(text) < 60:
        return 0.0
    hits = 0
    for pat in META_PHRASES:
        hits += len(re.findall(pat, text, re.I))
    if hits >= 4:
        return 1.0
    if hits >= 3:
        return 0.8
    if hits >= 2:
        return 0.6
    if hits >= 1:
        return 0.3
    return 0.0


def _score_no_imperfections(text: str) -> float:
    """Penalize text with zero human imperfections (typos, fragments, hedges, self-corrections)."""
    words = len(text.split())
    if words < 25:
        return 0.0
    sentences = _split_sentences(text)
    imperfection_signals = 0

    fragments = sum(1 for s in sentences if len(s.split()) <= 5)
    if fragments > 0:
        imperfection_signals += fragments

    hedges = len(re.findall(
        r"\b(?:I think|not sure|maybe|probably|might|could be|"
        r"I guess|feels like|seems like|hard to say|"
        r"not 100%|not totally|sorta|kinda)\b", text, re.I
    ))
    imperfection_signals += hedges

    self_corrections = len(re.findall(
        r"\b(?:well actually|wait no|I mean|or rather|"
        r"scratch that|actually wait)\b", text, re.I
    ))
    imperfection_signals += self_corrections * 2

    asides = len(re.findall(r"\((?:worth noting|like|tbh|though|fwiw|imo)", text, re.I))
    imperfection_signals += asides

    spacing_quirks = len(re.findall(r"\w ,\w", text))
    imperfection_signals += spacing_quirks

    ratio = imperfection_signals / max(len(sentences), 1)
    if ratio > 0.5:
        return 0.0
    if ratio > 0.3:
        return 0.2
    if ratio > 0.15:
        return 0.4
    if ratio > 0.05:
        return 0.6
    return 0.9


def _score_perfect_grammar(text: str) -> float:
    """High score = suspiciously perfect grammar. External detectors weight this heavily."""
    words = len(text.split())
    if words < 25:
        return 0.0
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 0.0

    all_complete = all(len(s.split()) >= 6 for s in sentences)
    all_end_properly = all(s.rstrip()[-1] in '.!?' for s in sentences if s.rstrip())
    no_fragments = all(len(s.split()) > 4 for s in sentences)

    has_comma_before_conj = bool(re.search(r',\s+(?:and|but|or|so)\s+', text))
    has_proper_lists = bool(re.search(r',\s+\w+,\s+and\s+', text))

    score = 0.0
    if all_complete:
        score += 0.3
    if all_end_properly:
        score += 0.2
    if no_fragments:
        score += 0.2
    if has_comma_before_conj and has_proper_lists:
        score += 0.15
    if not re.search(r'\.\s*\.\s', text):
        score += 0.15

    return min(score, 1.0)


def _clean_for_scoring(text: str) -> str:
    text = re.sub(r'^#+.*$', '', text, flags=re.M)
    text = re.sub(r'^---\s*$', '', text, flags=re.M)
    text = re.sub(r'^\*\*.*?\*\*\s*$', '', text, flags=re.M)
    text = re.sub(r'^Rating:.*$', '', text, flags=re.M)
    text = re.sub(r'^## RATING.*$', '', text, flags=re.M)
    text = re.sub(r'^## KEY AX.*$', '', text, flags=re.M)
    return text.strip()


def score_text(text: str) -> dict:
    """
    Full AI-likelihood score breakdown.
    Returns dict with per-signal scores and weighted overall (0-1).
    Calibrated to approximate GPTZero/ZeroGPT detection thresholds.
    """
    clean = _clean_for_scoring(text)
    if len(clean) < 50:
        return {"overall": 0.0, "detail": "Text too short to score"}

    tokens = _tokenize(clean)
    sentences = _split_sentences(clean)

    scores = {
        "ai_vocabulary":        _score_ai_vocabulary(tokens),
        "sentence_uniformity":  _score_sentence_uniformity(sentences),
        "transition_density":   _score_transition_density(sentences),
        "contraction_absence":  _score_contraction_absence(clean),
        "em_dash_usage":        _score_em_dash_usage(clean),
        "vocab_diversity":      _score_vocab_diversity(tokens),
        "paragraph_uniformity": _score_paragraph_uniformity(clean),
        "formality_level":      _score_formality(clean),
        "opener_repetition":    _score_opener_repetition(clean),
        "ai_phrasing":          _score_ai_phrasing(clean),
        "eval_cliches":         _score_eval_cliches(clean),
        "meta_phrases":         _score_meta_phrases(clean),
        "no_imperfections":     _score_no_imperfections(clean),
        "perfect_grammar":      _score_perfect_grammar(clean),
    }

    weights = {
        "ai_vocabulary":        0.14,
        "sentence_uniformity":  0.08,
        "transition_density":   0.04,
        "contraction_absence":  0.04,
        "em_dash_usage":        0.02,
        "vocab_diversity":      0.03,
        "paragraph_uniformity": 0.04,
        "formality_level":      0.08,
        "opener_repetition":    0.06,
        "ai_phrasing":          0.05,
        "eval_cliches":         0.12,
        "meta_phrases":         0.04,
        "no_imperfections":     0.12,
        "perfect_grammar":      0.14,
    }

    overall = sum(scores[k] * weights[k] for k in scores)
    scores["overall"] = round(overall, 3)
    return scores


def score_field(text: str) -> float:
    return score_text(text).get("overall", 0.0)


def format_score_report(scores: dict, label: str = "") -> str:
    lines = []
    if label:
        lines.append(f"  [{label}]")
    if isinstance(scores.get("detail"), str):
        lines.append(f"    {scores['detail']}")
        return "\n".join(lines)

    overall = scores["overall"]
    if overall < 0.25:
        verdict = "LIKELY HUMAN"
    elif overall < 0.45:
        verdict = "MIXED"
    elif overall < 0.65:
        verdict = "LIKELY AI"
    else:
        verdict = "STRONGLY AI"

    lines.append(f"    Overall: {overall:.1%} => {verdict}")
    for k, v in scores.items():
        if k in ("overall", "detail"):
            continue
        bar = "#" * int(v * 20) + "." * (20 - int(v * 20))
        lines.append(f"    {k:25s} [{bar}] {v:.0%}")
    return "\n".join(lines)


def check_pr_references(text: str) -> list[str]:
    issues = []
    t = text.lower()
    if re.search(r'#\d{3,6}', text):
        issues.append("PR number reference (#digits)")
    if re.search(r'pull/\d+', t):
        issues.append("Pull URL reference")
    if re.search(r'\bpr[\s-]?\d+', t):
        issues.append("PR label reference")
    for phrase in ["this pr", "the pr", "this pull request", "the pull request"]:
        if re.search(r'\b' + re.escape(phrase) + r'\b', t):
            issues.append(f'PR phrase: "{phrase}"')
    return issues


def check_role_prompting(text: str) -> list[str]:
    issues = []
    patterns = [
        (r'(?i)\byou are a[n]?\s+\w+', "Role assignment"),
        (r'(?i)\bact as a[n]?\s+\w+', "Role instruction"),
        (r'(?i)\bas a senior\b', "Seniority reference"),
    ]
    for pat, label in patterns:
        if re.search(pat, text):
            issues.append(label)
    return issues


def check_em_dashes(text: str) -> list[str]:
    issues = []
    if "\u2014" in text:
        issues.append(f"Em-dash found {text.count(chr(0x2014))}x")
    if "\u2013" in text:
        issues.append(f"En-dash found {text.count(chr(0x2013))}x")
    dd = len(re.findall(r'(?<!\-)\-\-(?!\-)', text))
    if dd:
        issues.append(f"Double-dash found {dd}x")
    return issues


def full_validation(text: str) -> dict:
    scores = score_text(text)
    return {
        "scores": scores,
        "overall": scores.get("overall", 0.0),
        "pr_issues": check_pr_references(text),
        "role_issues": check_role_prompting(text),
        "dash_issues": check_em_dashes(text),
        "passed": (
            scores.get("overall", 1.0) < 0.30
            and not check_pr_references(text)
            and not check_role_prompting(text)
            and not check_em_dashes(text)
        ),
    }
