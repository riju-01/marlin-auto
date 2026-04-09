# Marlin V3 Automation

Generate humanized feedback answers for HFI (Human Feedback Interface).

See [docs/README.md](docs/README.md) for full documentation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env with your API key

# 3. Run
python marlin_auto.py
```

## Project Structure

```
marlin-auto/
├── marlin_auto.py           # Entry point — main orchestrator
├── core/                    # Configuration, LLM client, utilities, AI scoring
├── pipeline/                # Feedback generation, humanization, prompt generation
├── integrations/            # GitHub PR fetcher, HFI session watcher
├── ui/                      # Rich terminal UI
├── tools/                   # Standalone CLI utilities (regen_feedback)
├── docs/                    # Documentation (README, HFI questions, notes)
└── tasks/                   # Task data (gitignored)
```
