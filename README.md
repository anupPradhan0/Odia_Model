# Odia Synthetic Dataset Generator

This project generates a synthetic Odia OCR dataset for fine-tuning TrOCR. It scrapes Odia text from Wikipedia, cleans it into sentences, renders those sentences into images using Odia-capable fonts, applies mild augmentations, and writes train and test labels automatically.

## Files

- `generate_odia_data.py`: main generation script
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `fonts/` directory and add Odia `.ttf` fonts such as:

- Baloo Bhaina 2
- Noto Sans Oriya
- Utkal

The script validates each font and skips any font that cannot render a known Odia sample string.

## Usage

Preview 10 images first:

```bash
python generate_odia_data.py --preview
```

If your environment supports image viewers, you can also try:

```bash
python generate_odia_data.py --preview --show-preview
```

Generate the full dataset:

```bash
python generate_odia_data.py
```

Optional overrides:

```bash
python generate_odia_data.py --num-images 5000 --fonts-folder ./fonts --output-folder ./odia_dataset
```

## Output Structure

```text
odia_dataset/
├── train/
├── test/
├── train_labels.csv
├── test_labels.csv
└── odia_sentences.txt
```

The `train_labels.csv` and `test_labels.csv` files contain:

- `image_name`: relative image path like `train/img_00001.jpg`
- `text`: exact Odia text rendered into that image

## Notes

- The script uses sentence-level train/test splitting to reduce text leakage across splits.
- Preview mode writes files into `odia_dataset/preview/` so quality can be checked before large runs.
- If Wikipedia scraping returns too little text or fails, the script falls back to a small built-in Odia sentence list so the pipeline remains testable.
- Pillow rendering is used as requested, but Odia shaping quality depends on the selected fonts and Pillow's rendering behavior. Always inspect preview output before generating a large dataset.# Odia_Model
