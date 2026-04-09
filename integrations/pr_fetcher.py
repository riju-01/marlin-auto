"""
Fetch and rank GitHub PRs for Marlin V3 suitability.

Uses GitHub REST API via Python requests (runs on Windows directly,
no WSL needed). Scores PRs on diff size, language, complexity, test coverage.
"""

import os
import re

import requests

from core.config import PR_DIFF_MIN_LINES, PR_DIFF_MAX_LINES, PR_DIFF_SWEET_SPOT, SUPPORTED_LANGUAGES


def _github_headers() -> dict:
    """Build headers with optional GitHub token for higher rate limits."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _get_json(url: str, timeout: int = 30) -> dict | list | None:
    """Fetch JSON from GitHub API using requests (Windows-native)."""
    try:
        resp = requests.get(url, timeout=timeout, headers=_github_headers())
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            if remaining == "0":
                import time
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset - int(time.time()), 0)
                print(f"  [rate limit] GitHub API rate limited. Resets in {wait}s.")
                print(f"  [rate limit] Add GITHUB_TOKEN to .env to avoid this.")
        return None
    except Exception:
        return None


def _get_text(url: str, timeout: int = 60) -> str:
    """Fetch raw text from a URL using requests (Windows-native)."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        return ""
    except Exception:
        return ""


def parse_pr_url(url: str) -> dict | None:
    """Extract owner, repo, number from a GitHub PR URL."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url.strip())
    if not m:
        return None
    return {"owner": m.group(1), "repo": m.group(2), "number": int(m.group(3))}


def fetch_pr_metadata(owner: str, repo: str, number: int) -> dict | None:
    """Fetch PR metadata from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    data = _get_json(url)
    if not data or "title" not in data:
        return None

    return {
        "title": data.get("title", ""),
        "body": (data.get("body") or "")[:3000],
        "state": data.get("state", ""),
        "merged_at": data.get("merged_at", ""),
        "author": data.get("user", {}).get("login", ""),
        "additions": data.get("additions", 0),
        "deletions": data.get("deletions", 0),
        "changed_files": data.get("changed_files", 0),
        "base_sha": data.get("base", {}).get("sha", ""),
        "base_ref": data.get("base", {}).get("ref", ""),
        "head_sha": data.get("head", {}).get("sha", ""),
        "diff_lines": data.get("additions", 0) + data.get("deletions", 0),
    }


def fetch_pr_files(owner: str, repo: str, number: int) -> list[dict]:
    """Fetch list of changed files."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files?per_page=100"
    data = _get_json(url)
    if not data or not isinstance(data, list):
        return []
    return [
        {
            "filename": f.get("filename", ""),
            "status": f.get("status", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
        }
        for f in data
    ]


def fetch_pr_diff(owner: str, repo: str, number: int) -> str:
    """Fetch the full PR diff."""
    url = f"https://github.com/{owner}/{repo}/pull/{number}.diff"
    return _get_text(url, timeout=60)


def fetch_repo_info(owner: str, repo: str) -> dict:
    """Fetch basic repo metadata."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data = _get_json(url)
    if not data:
        return {}
    return {
        "description": data.get("description", ""),
        "language": (data.get("language") or "").lower(),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "size_kb": data.get("size", 0),
        "archived": data.get("archived", False),
        "topics": data.get("topics", []),
    }


def score_pr(pr_meta: dict, files: list[dict], repo_info: dict) -> dict:
    """
    Score a PR for Marlin suitability (0-100).
    Returns score breakdown with reasons.
    """
    score = 0
    reasons = []

    diff_lines = pr_meta.get("diff_lines", 0)
    if PR_DIFF_SWEET_SPOT[0] <= diff_lines <= PR_DIFF_SWEET_SPOT[1]:
        score += 25
        reasons.append(f"Diff size {diff_lines} lines (sweet spot)")
    elif PR_DIFF_MIN_LINES <= diff_lines <= PR_DIFF_MAX_LINES:
        score += 15
        reasons.append(f"Diff size {diff_lines} lines (acceptable)")
    elif diff_lines < PR_DIFF_MIN_LINES:
        score += 5
        reasons.append(f"Diff size {diff_lines} lines (too small)")
    else:
        score += 5
        reasons.append(f"Diff size {diff_lines} lines (too large)")

    lang = repo_info.get("language", "")
    if lang in SUPPORTED_LANGUAGES:
        score += 15
        reasons.append(f"Language: {lang} (supported)")
    else:
        score += 2
        reasons.append(f"Language: {lang} (check support)")

    test_files = [f for f in files if "test" in f["filename"].lower()]
    if len(test_files) >= 3:
        score += 15
        reasons.append(f"{len(test_files)} test files changed (great)")
    elif test_files:
        score += 10
        reasons.append(f"{len(test_files)} test file(s) changed")
    else:
        score += 0
        reasons.append("No test files in diff (risky)")

    unique_dirs = set()
    for f in files:
        parts = f["filename"].split("/")
        if len(parts) >= 2:
            unique_dirs.add(parts[0])
    if len(unique_dirs) >= 3:
        score += 15
        reasons.append(f"Cross-module: {len(unique_dirs)} top-level dirs")
    elif len(unique_dirs) >= 2:
        score += 10
        reasons.append(f"{len(unique_dirs)} top-level dirs touched")
    else:
        score += 3
        reasons.append("Single directory (less complex)")

    if pr_meta.get("merged_at"):
        score += 10
        reasons.append("PR is merged (has known-good answer)")
    else:
        score += 3
        reasons.append("PR not merged yet")

    file_count = pr_meta.get("changed_files", 0)
    if 5 <= file_count <= 20:
        score += 10
        reasons.append(f"{file_count} files changed (good scope)")
    elif file_count > 20:
        score += 5
        reasons.append(f"{file_count} files (large scope)")
    else:
        score += 3
        reasons.append(f"{file_count} files (narrow scope)")

    stars = repo_info.get("stars", 0)
    if 1000 <= stars <= 30000:
        score += 10
        reasons.append(f"{stars:,} stars (low memorization risk)")
    elif stars > 80000:
        score += 2
        reasons.append(f"{stars:,} stars (high memorization risk)")
    else:
        score += 6

    return {
        "score": min(score, 100),
        "reasons": reasons,
    }


def fetch_and_rank_pr(url: str) -> dict | None:
    """Fetch a single PR and return all data needed for ranking."""
    parsed = parse_pr_url(url)
    if not parsed:
        return None

    owner, repo, number = parsed["owner"], parsed["repo"], parsed["number"]
    pr_meta = fetch_pr_metadata(owner, repo, number)
    if not pr_meta:
        return None

    files = fetch_pr_files(owner, repo, number)
    repo_info = fetch_repo_info(owner, repo)
    ranking = score_pr(pr_meta, files, repo_info)

    return {
        "url": url.strip(),
        "owner": owner,
        "repo": repo,
        "number": number,
        "meta": pr_meta,
        "files": files,
        "repo_info": repo_info,
        "ranking": ranking,
        "task_name": f"{owner.lower()}_{repo}_{number}",
    }


def rank_prs(urls: list[str], status_callback=None) -> list[dict]:
    """
    Fetch and rank multiple PRs.
    status_callback(msg) is called with progress updates.
    """
    results = []
    for i, url in enumerate(urls):
        if status_callback:
            status_callback(f"Fetching PR {i+1}/{len(urls)}: {url.strip()}")
        pr_data = fetch_and_rank_pr(url)
        if pr_data:
            results.append(pr_data)
        elif status_callback:
            status_callback(f"  Failed to fetch: {url.strip()}")

    results.sort(key=lambda x: x["ranking"]["score"], reverse=True)
    return results
