# Kronos-demo — Claude Instructions

## Project Overview
Offline backtesting and research tool that evaluates the performance of the Kronos time-series forecasting model. The evaluation is performed on BTC/USDT candlestick data from Binance. It runs
the Kronos time-series forecasting model, and generates common trading performance indicators. It is **not** a live-deployment or auto-publishing system.

## Project Structure
```
update_predictions.py   # Main entry point — run once manually
model/                  # Kronos model source (kronos.py, module.py, __init__.py)
index.html              # Output dashboard (updated in-place)
prediction_chart.png    # Output chart (overwritten each run)
style.css               # Dashboard stylesheet
img/                    # Static assets (logo, etc.)
requirements.txt        # Pinned dependencies
venv/                   # Local Python 3.11 virtual environment
../Kronos_model/        # HuggingFace model cache (outside repo, do NOT modify)
```

## How to Run
```bash
venv/Scripts/python update_predictions.py
```
Always use the local **venv** — never system Python.

## Installing / Updating Dependencies
```bash
venv/Scripts/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
```
`torch==2.10.0+cu128` requires the PyTorch CUDA index; other packages resolve from PyPI.

## Environment
- **Python**: 3.11 via `venv/Scripts/python`
- **GPU**: NVIDIA GeForce RTX 3090, CUDA 12.8 (`torch==2.10.0+cu128`)
- **Model cache**: `../Kronos_model/` — base weights from `NeoQuasar/Kronos-mini` plus
  locally fine-tuned checkpoints. The cache directory is outside the repo root.

## Hard Rules — Never Do Without Explicit Confirmation
- **No auto-commit or git push** — never run `git commit` or `git push` automatically.
- **No touching model weights** — never modify, delete, or overwrite anything in `../Kronos_model/`.
- **No touching secrets** — never read or modify `.env` files or API credentials.
- **Never enable the scheduler** — `run_scheduler(loaded_model)` at the bottom of
  `update_predictions.py` must remain commented out unless explicitly asked to enable it.

## Coding Conventions
- **English only** — all code, comments, print statements, and commit messages in English.
- **Minimal changes** — only modify what is directly requested; do not refactor, clean up, or
  reorganize surrounding code.
- No type annotations, docstrings, or comments added to code that wasn't changed.
