"""
AI detection scoring engine.

Scores text on multiple signals that AI detectors look for.
Each sub-score is 0-1 (0 = human, 1 = AI). Weighted average gives overall score.
Target: keep overall below 0.30 (30%).
"""

import math
import re
from collections import Counter

AI_VOCABULARY = {
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
}

AI_TRANSITIONS = {
    "however", "therefore", "thus", "moreover", "furthermore",
    "additionally", "consequently", "nevertheless", "nonetheless",
    "specifically", "notably", "essentially", "fundamentally",
    "ultimately", "importantly", "significantly", "interestingly",
    "crucially", "overall",
}


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
    if cv < 0.35:
        return 0.7
    if cv < 0.5:
        return 0.3
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
    contractions = len(re.findall(r"\b\w+n't\b|\b\w+'(?:ve|re|ll|d|s|m)\b", text))
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
    informal = len(re.findall(
        r"\b(?:gonna|gotta|kinda|sorta|y'all|dunno|wanna|yeah|nah|ok|okay)\b", text, re.I
    ))
    dashes_as_pauses = len(re.findall(r" - ", text))
    parens = len(re.findall(r"\(", text))
    words = len(text.split())
    if words < 20:
        return 0.0
    human_signals = contractions + informal + dashes_as_pauses + parens
    ratio = human_signals / words
    if ratio > 0.03:
        return 0.0
    if ratio > 0.015:
        return 0.2
    if ratio > 0.005:
        return 0.5
    return 0.9


def _score_opener_repetition(text: str) -> float:
    """Detect repeated sentence/paragraph openers - a strong AI signal."""
    sentences = _split_sentences(text)
    if len(sentences) < 4:
        return 0.0
    openers = []
    for s in sentences:
        words = s.strip().split()[:3]
        openers.append(" ".join(w.lower() for w in words))
    counts = Counter(openers)
    if not counts:
        return 0.0
    most_common_count = counts.most_common(1)[0][1]
    ratio = most_common_count / len(openers)
    if ratio > 0.4:
        return 1.0
    if ratio > 0.25:
        return 0.6
    if ratio > 0.15:
        return 0.3
    return 0.0


def _clean_for_scoring(text: str) -> str:
    """Strip markdown headers, metadata lines, and rating lines before scoring."""
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
    }

    weights = {
        "ai_vocabulary":        0.18,
        "sentence_uniformity":  0.10,
        "transition_density":   0.13,
        "contraction_absence":  0.08,
        "em_dash_usage":        0.05,
        "vocab_diversity":      0.08,
        "paragraph_uniformity": 0.13,
        "formality_level":      0.12,
        "opener_repetition":    0.13,
    }

    overall = sum(scores[k] * weights[k] for k in scores)
    scores["overall"] = round(overall, 3)
    return scores


def score_field(text: str) -> float:
    """Quick single-number AI score for one feedback field."""
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
    """Check for leaked PR references that would cause rejection."""
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
    """Check for role-based prompting patterns."""
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
    """Run all checks: AI score + PR refs + role prompting + em dashes."""
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
