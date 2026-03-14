# Odia Model — Read & Understand Odia Text

**Goal:** Build tools so machines can **read** Odia (ଓଡ଼ିଆ) the way people do — from real, human-written Odia sentences and documents — and turn that into text a computer can work with (and later, understand).

This repo is a practical first step: **synthetic Odia OCR data** plus **TrOCR fine-tuning**, so a model learns to recognize Odia script in images. That’s the bridge from “pixels of Odia” to “Unicode Odia text” you can search, translate, or feed into larger language models.

## What this project does

| Piece | Purpose |
|--------|--------|
| **Dataset generator** | Pulls Odia text (e.g. from Wikipedia), cleans it into sentences, renders it with Odia fonts, augments lightly, and writes **image + label** pairs for training. |
| **TrOCR fine-tuning** | Trains a text-recognition model on those pairs so it can **read Odia from images** (screenshots, scans, rendered pages). |
| **Validation** | Checks that labels match generated images before you train. |

Longer-term, the same pipeline supports anything built on **human-readable Odia**: OCR for books and web, accessibility, and downstream NLP once text is extracted.

## Repository layout

- `scripts/` — runnable Python scripts (generate data, validate, fine-tune)
- `assets/fonts/` — Odia-capable `.ttf` fonts used when rendering text to images
- `docs/` — notes and planning
- `odia_dataset/` — generated dataset (train/test images + CSV labels)
- `saved_model/` — fine-tuned TrOCR checkpoint (after training)
- `requirements.txt` — Python dependencies

## Setup

1. Use **Python 3.10+** and a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put Odia fonts in `assets/fonts/`, for example:

   - Baloo Bhaina 2  
   - Noto Sans Oriya  
   - Utkal  

   The generator checks each font and skips fonts that can’t render a known Odia sample.

## Usage

**Preview** (small run, check quality first):

```bash
python scripts/generate_odia_data.py --preview
```

Optional: open previews if your environment has a viewer:

```bash
python scripts/generate_odia_data.py --preview --show-preview
```

**Full dataset:**

```bash
python scripts/generate_odia_data.py
```

**Overrides:**

```bash
python scripts/generate_odia_data.py --num-images 5000 --fonts-folder ./assets/fonts --output-folder ./odia_dataset
```

## Dataset output

```text
odia_dataset/
├── train/
├── test/
├── train_labels.csv
├── test_labels.csv
└── odia_sentences.txt
```

`train_labels.csv` / `test_labels.csv` columns:

- `image_name` — e.g. `train/img_00001.jpg`
- `text` — exact Odia string rendered in that image

## Validate labels

```bash
python scripts/validate_labels.py
```

Optional JSON report:

```bash
python scripts/validate_labels.py --report-json validation_report.json
```

## Fine-tune TrOCR (GPU)

Install CUDA PyTorch in the same venv (pick the wheel that matches your CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Train:

```bash
python scripts/finetune_trocr.py
```

Best checkpoint by test **CER** is saved under `saved_model/`.

## Notes

- Train/test split is **sentence-level** to limit text leakage between splits.
- Preview files go to `odia_dataset/preview/` so you can judge font/shaping before big runs.
- If Wikipedia scraping is thin or fails, the script falls back to a small built-in Odia sentence list so the pipeline still runs.
- Rendering quality depends on **fonts** and **Pillow**; always inspect previews before large generations.

## Roadmap (idea)

- More diverse **human Odia** sources (news, books, user contributions).  
- Stronger OCR + optional **Odia NLP** (tokenization, summarization, Q&A) on top of recognized text.

---

*Contributions welcome — especially more Odia text sources and evaluation on real-world Odia images.*
