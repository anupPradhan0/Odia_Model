# Prompt — Fine-tune TrOCR on Odia Dataset

## System
- OS: Arch Linux (Omarchy) — use `pacman` and Python venv
- GPU: NVIDIA RTX 3050 Mobile — use CUDA
- No conda, no system-wide pip installs

## Setup (run this first)
```bash
# Install Python and CUDA — try yay first, fallback to pacman
yay -S python cuda || sudo pacman -S python cuda

# Create and activate virtual environment
python -m venv odia_env
source odia_env/bin/activate

# Install Python libraries inside venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow pandas jiwer tqdm
```

## Task
Build `finetune_trocr.py` that fine-tunes TrOCR on the Odia dataset.

## Context
- Dataset lives in `odia_dataset/`
- `train_labels.csv` and `test_labels.csv` — columns: `image_name`, `text`
- Always activate `odia_env` before running

## What the script must do
1. Load both CSV files using a custom PyTorch Dataset class
2. Load `microsoft/trocr-base-handwritten` from HuggingFace
3. Fine-tune for 10 epochs on train images using CUDA
4. Print epoch, training loss, and CER score after each epoch
5. Save the best model (lowest CER) to `saved_model/`
6. Batch size 8, learning rate 5e-5, AdamW optimizer

## Expected usage
```bash
source odia_env/bin/activate
python finetune_trocr.py
```