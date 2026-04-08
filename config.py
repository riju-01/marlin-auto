"""Central configuration for Marlin V3 Automation."""

import os
import sys

_INSIDE_WSL = sys.platform.startswith("linux")

# ---------------------------------------------------------------------------
# Paths  (auto-detect Windows vs WSL)
# ---------------------------------------------------------------------------

if _INSIDE_WSL:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    WORKSPACE = os.path.dirname(_script_dir)           # parent of marlin-auto
    DATA_DIR = os.path.join(_script_dir, "tasks")
    MARLIN_V2 = os.path.join(WORKSPACE, "marlin-v2")
    SCRIPTS_DIR = os.path.join(MARLIN_V2, "scripts")
    TEAMMATE_PROMPT_PATH = os.path.join(
        MARLIN_V2, ".cursor", "rules", "TEAMMATE_PROMPT.md"
    )
    EVALUATE_RULES_PATH = os.path.join(
        MARLIN_V2, ".cursor", "rules", "marlin-evaluate.mdc"
    )
    FEEDBACK_TEMPLATE_PATH = os.path.join(
        MARLIN_V2, "templates", "feedback_template.md"
    )
    WIN_WORKSPACE = None
else:
    WIN_WORKSPACE = os.environ.get(
        "MARLIN_WORKSPACE", r"C:\Users\RIJU\Desktop\cognyzer\marlin"
    )
    WORKSPACE = "/mnt/c/" + WIN_WORKSPACE[3:].replace("\\", "/")

    WIN_MARLIN_AUTO = os.path.join(WIN_WORKSPACE, "marlin-auto")
    DATA_DIR = os.path.join(WIN_MARLIN_AUTO, "tasks")

    MARLIN_V2 = f"{WORKSPACE}/marlin-v2"
    SCRIPTS_DIR = f"{MARLIN_V2}/scripts"

    TEAMMATE_PROMPT_PATH = os.path.join(
        WIN_WORKSPACE, "marlin-v2", ".cursor", "rules", "TEAMMATE_PROMPT.md"
    )
    EVALUATE_RULES_PATH = os.path.join(
        WIN_WORKSPACE, "marlin-v2", ".cursor", "rules", "marlin-evaluate.mdc"
    )
    FEEDBACK_TEMPLATE_PATH = os.path.join(
        WIN_WORKSPACE, "marlin-v2", "templates", "feedback_template.md"
    )

# WSL path to this package (for locating optional bin/claude-hfi)
if _INSIDE_WSL:
    MARLIN_AUTO_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    MARLIN_AUTO_DIR = f"{WORKSPACE}/marlin-auto"

# Put your claude-hfi binary here (gitignored). Phase 3 copies it into each task repo.
HFI_LOCAL_DIR = f"{MARLIN_AUTO_DIR}/HFI"

WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu")

# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_FALLBACK = "gemini-2.0-flash-lite"

AI_SCORE_TARGET = 0.30
MAX_HUMANIZE_PASSES = 6
GEMINI_RATE_LIMIT_SLEEP = 4

# ---------------------------------------------------------------------------
# HFI settings
# ---------------------------------------------------------------------------

HFI_POLL_INTERVAL = 10
HFI_POLL_TIMEOUT = 1800
INTERFACE_CODE = "cc_agentic_coding_next"

# ---------------------------------------------------------------------------
# PR ranking
# ---------------------------------------------------------------------------

PR_DIFF_MIN_LINES = 200
PR_DIFF_MAX_LINES = 800
PR_DIFF_SWEET_SPOT = (300, 600)

SUPPORTED_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "java", "c++", "c",
    "ruby", "php", "scala", "swift", "kotlin", "dart", "elixir",
}
