# Cursor Agent Prompt — Odia Synthetic Handwriting Dataset Generator

## Project Overview

I am building a **Multimodal AI system for the Odia language** called an "Intelligent Odia Vision System." The goal is to create an AI that can:
- Read both printed and handwritten Odia script from images
- Understand the meaning and context of Odia documents
- Answer questions and summarise Odia documents in the Odia language

The **first and most critical step** is generating a large synthetic training dataset of Odia handwriting images. This file describes exactly what I need you to build.

---

## What I Need You to Build

A Python script (or Jupyter notebook) that **automatically generates thousands of synthetic Odia handwriting images** along with their labels, to be used for fine-tuning a TrOCR model.

---

## Tech Stack

- **Language**: Python 3.10+
- **Libraries**:
  - `Pillow` — image creation and manipulation
  - `wikipedia-api` — scraping Odia text from Wikipedia
  - `numpy` — array operations
  - `imgaug` or `albumentations` — image augmentation
  - `pandas` — saving labels to CSV
  - `tqdm` — progress bars
- **Font files**: `.ttf` Odia font files (Baloo Bhaina 2, Noto Sans Oriya, Utkal) — script should accept a folder of font files

---

## Full Pipeline — Step by Step

### Step 1 — Collect Odia Text
- Use the `wikipediaapi` Python library to scrape Odia language Wikipedia articles
- Extract clean sentences from those articles
- Clean the text: remove English characters, special symbols, extra spaces
- Save all sentences to a `odia_sentences.txt` file
- Target: minimum **10,000 unique Odia sentences**

### Step 2 — Render Text onto Images
- For each sentence, use `Pillow` (PIL) to:
  - Create a white or off-white background image
  - Write the Odia text onto the image using a randomly selected Odia `.ttf` font
  - Randomly vary: font size (between 18px and 36px), text position, line spacing
  - Use multiple font files so not all images look the same

### Step 3 — Apply Augmentations (make fake images look real)
Apply the following augmentations **randomly** to each image to simulate real handwriting conditions:

| Augmentation | Details |
|---|---|
| Rotation / tilt | Random rotation between -15 and +15 degrees |
| Gaussian blur | Slight blur to simulate pen ink spread |
| Random noise | Salt and pepper noise to simulate paper grain |
| Brightness/contrast jitter | Simulate different lighting conditions |
| Paper texture overlay | Optional: overlay a subtle lined or yellowed paper texture |
| Elastic distortion | Slight warping to simulate natural handwriting variation |

Each image should have **2-3 augmentations applied randomly** — not all at once every time.

### Step 4 — Auto-generate Labels
- For every image generated, automatically save a row in `labels.csv` with:
  - `image_name` — filename of the image (e.g. `img_00001.jpg`)
  - `text` — the exact Odia text that was rendered onto that image
- No manual labelling is needed — the script knows what it rendered

### Step 5 — Train/Test Split
- After generating all images, automatically split into:
  - `train/` folder — 80% of images
  - `test/` folder — 20% of images
- Generate separate `train_labels.csv` and `test_labels.csv`

---

## Output Folder Structure

```
odia_dataset/
├── train/
│   ├── img_00001.jpg
│   ├── img_00002.jpg
│   └── ...
├── test/
│   ├── img_08001.jpg
│   └── ...
├── train_labels.csv
├── test_labels.csv
└── odia_sentences.txt
```

---

## Configuration

At the top of the script, expose these as easy-to-change variables:

```python
FONTS_FOLDER = "./fonts"         # folder containing .ttf Odia font files
OUTPUT_FOLDER = "./odia_dataset" # where to save generated images
NUM_IMAGES = 10000               # total number of images to generate
IMAGE_WIDTH = 512                # width of each image in pixels
IMAGE_HEIGHT = 64                # height of each image in pixels
FONT_SIZE_MIN = 18               # minimum font size
FONT_SIZE_MAX = 36               # maximum font size
TRAIN_SPLIT = 0.8                # 80% train, 20% test
AUGMENTATION_PROB = 0.7          # probability of applying augmentation to each image
```

---

## Important Notes for the Agent

1. **Odia Unicode range is U+0B00 to U+0B7F** — make sure text rendering supports this range correctly with Pillow
2. **Font files must support Odia script** — if a font doesn't support Odia, skip it and log a warning
3. **Add a progress bar** using `tqdm` so I can see how many images have been generated
4. **Handle errors gracefully** — if a sentence fails to render (e.g. font doesn't support a character), skip it and continue
5. **Add a `--preview` flag** — when run with `python generate.py --preview`, generate only 10 sample images and display them so I can check quality before running the full generation
6. **Log statistics at the end** — total images generated, total sentences used, time taken, average image size

---

## How to Run (expected usage)

```bash
# Install dependencies
pip install pillow wikipedia-api numpy imgaug albumentations pandas tqdm

# Preview 10 sample images first
python generate_odia_data.py --preview

# Generate full dataset
python generate_odia_data.py
```

---

## Context — Why This Dataset is Being Built

This dataset will be used to **fine-tune Microsoft's TrOCR model** (specifically `microsoft/trocr-base-handwritten` from HuggingFace) on Odia script. TrOCR is a transformer-based OCR model that takes an image as input and outputs the text in that image. Fine-tuning it on synthetic Odia data will teach it to read Odia handwriting.

After fine-tuning TrOCR, the extracted Odia text will be passed to **IndicBERT** (by AI4Bharat) for language understanding, and then to **Qwen-VL** for question answering and summarisation — all in Odia.

The synthetic dataset is the foundation of this entire pipeline.

---

## Deliverables Expected from You (Cursor)

1. `generate_odia_data.py` — the main generation script
2. `requirements.txt` — all dependencies
3. A short `README.md` explaining how to run it and what fonts to download