"""Core utilities: configuration, LLM client, helpers, and AI scoring."""

from core.config import *  # noqa: F401,F403
from core.llm_client import generate, detect_provider, provider_info, get_model
from core.utils import (
    wsl_exec, wsl_exec_script, read_wsl_file, write_wsl_file,
    ensure_task_dir, write_task_file, read_task_file, load_json,
    curl_json, curl_text,
)
from core.ai_scorer import score_field, score_text, full_validation, format_score_report
