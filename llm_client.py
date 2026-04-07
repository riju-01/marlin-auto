"""
Unified LLM client that supports Gemini, OpenAI, and Claude.

Auto-detects which provider to use based on which API key is set in .env.
All modules call llm_client.generate() instead of provider-specific code.

Priority if multiple keys set: Claude > OpenAI > Gemini
"""

import os
import time
from pathlib import Path

import requests

PROVIDERS = {
    "claude": {
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
        "fallback_model": "claude-haiku-4-20250514",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "fallback_model": "gpt-4o-mini",
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
        "fallback_model": "gemini-2.0-flash-lite",
    },
}


def _load_env():
    """Load .env file from the marlin-auto directory."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


_load_env()


def detect_provider() -> tuple[str, str]:
    """
    Detect which LLM provider to use based on available API keys.
    Returns (provider_name, api_key).
    """
    for provider_name in ["claude", "openai", "gemini"]:
        env_key = PROVIDERS[provider_name]["env_key"]
        api_key = os.environ.get(env_key, "").strip()
        if api_key:
            return provider_name, api_key
    return "", ""


def get_model(provider: str) -> str:
    """Get the model name, allowing override via LLM_MODEL env var."""
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override
    return PROVIDERS.get(provider, {}).get("default_model", "")


def get_fallback_model(provider: str) -> str:
    return PROVIDERS.get(provider, {}).get("fallback_model", "")


def generate(prompt: str, api_key: str = None, provider: str = None,
             model: str = None, max_retries: int = 3,
             timeout: int = 120) -> str | None:
    """
    Generate text from any supported LLM provider.
    Auto-detects provider if not specified.
    Returns generated text or None on failure.
    """
    if not provider or not api_key:
        provider, api_key = detect_provider()

    if not provider or not api_key:
        return None

    if not model:
        model = get_model(provider)

    for attempt in range(max_retries):
        result = _call_provider(provider, api_key, model, prompt, timeout)
        if result is not None:
            return result

        if attempt < max_retries - 1:
            fallback = get_fallback_model(provider)
            if fallback and fallback != model:
                result = _call_provider(provider, api_key, fallback, prompt, timeout)
                if result is not None:
                    return result

            wait = min(30, 4 * (attempt + 1))
            time.sleep(wait)

    return None


def _call_provider(provider: str, api_key: str, model: str,
                   prompt: str, timeout: int) -> str | None:
    if provider == "gemini":
        return _call_gemini(api_key, model, prompt, timeout)
    elif provider == "openai":
        return _call_openai(api_key, model, prompt, timeout)
    elif provider == "claude":
        return _call_claude(api_key, model, prompt, timeout)
    return None


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _call_gemini(api_key: str, model: str, prompt: str, timeout: int) -> str | None:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
        if resp.status_code == 429:
            time.sleep(5)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _call_openai(api_key: str, model: str, prompt: str, timeout: int) -> str | None:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.8,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        if resp.status_code == 429:
            time.sleep(5)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------

def _call_claude(api_key: str, model: str, prompt: str, timeout: int) -> str | None:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("content", [])
            if content:
                return content[0].get("text", "").strip()
        if resp.status_code == 429:
            time.sleep(5)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def provider_info() -> dict:
    """Return info about the detected provider for display."""
    provider, api_key = detect_provider()
    if not provider:
        return {"provider": "none", "model": "", "status": "No API key found in .env"}

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    return {
        "provider": provider,
        "model": get_model(provider),
        "key_preview": masked_key,
        "status": "ready",
    }
