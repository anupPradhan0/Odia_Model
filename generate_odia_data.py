from __future__ import annotations

import argparse
import logging
import math
import random
import re
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import wikipediaapi
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm


FONTS_FOLDER = "./fonts"
OUTPUT_FOLDER = "./odia_dataset"
NUM_IMAGES = 10000
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 64
FONT_SIZE_MIN = 18
FONT_SIZE_MAX = 36
TRAIN_SPLIT = 0.8
AUGMENTATION_PROB = 0.7
IMAGE_EXTENSION = ".jpg"
JPEG_QUALITY = 95
PREVIEW_COUNT = 10
WIKIPEDIA_TARGET_SENTENCES = 10000
WIKIPEDIA_MAX_PAGES = 250
RANDOM_SEED = 42

ODIA_SAMPLE_TEXT = "ନମସ୍କାର ଓଡ଼ିଆ ଭାଷା"
DEFAULT_SEED_TITLES = [
    "ଓଡ଼ିଆ ଭାଷା",
    "ଓଡ଼ିଶା",
    "ଭୁବନେଶ୍ୱର",
    "କଟକ",
    "ପୁରୀ",
    "ରବୀନ୍ଦ୍ରନାଥ ଠାକୁର",
    "ଜଗନ୍ନାଥ ମନ୍ଦିର",
    "ଭାରତ",
    "ଶିକ୍ଷା",
    "ସାହିତ୍ୟ",
]
FALLBACK_SENTENCES = [
    "ଏହା ଏକ ନମୁନା ଓଡ଼ିଆ ବାକ୍ୟ ଅଟେ।",
    "ଓଡ଼ିଆ ଭାଷା ଭାରତର ଏକ ପ୍ରମୁଖ ଭାଷା।",
    "ଏହି ତଥ୍ୟସମୂହ ଲିପି ପରିଚୟ ଶିକ୍ଷଣ ପାଇଁ ବ୍ୟବହୃତ ହେବ।",
    "ମାନବ ଲେଖା ଭଳି ଦେଖାଯାଉଥିବା ଛବି ତିଆରି କରାଯାଉଛି।",
    "ମଡେଲ୍ ଟିକୁ ନୂତନ ଓଡ଼ିଆ ଲିପି ଶିଖାଯାଉଛି।",
    "ପ୍ରତ୍ୟେକ ଛବି ପାଇଁ ସଠିକ ଟେକ୍ସଟ୍ ଲେବେଲ୍ ସଂରକ୍ଷିତ ହେବ।",
    "ଏହି ପ୍ରକ୍ରିୟାରେ ଛବିଗୁଡ଼ିକୁ ଘୁରାଇ ଏବଂ ଧୁସର କରାଯାଇପାରେ।",
    "ଶେଷରେ ତଥ୍ୟସମୂହକୁ ଟ୍ରେନ୍ ଏବଂ ଟେଷ୍ଟ ଭାଗରେ ବିଭକ୍ତ କରାଯିବ।",
    "ଓଡ଼ିଆ ଉଇକିପିଡ଼ିଆରୁ ବାକ୍ୟ ସଂଗ୍ରହ କରିବା ଉଦ୍ଦେଶ୍ୟ ଅଟେ।",
    "ପୂର୍ବଦର୍ଶନ ମୋଡ୍ ରେ କେବଳ କିଛି ନମୁନା ତିଆରି କରାଯିବ।",
]

ODIA_CHAR_PATTERN = re.compile(r"[\u0B00-\u0B7F]")
ALLOWED_TEXT_PATTERN = re.compile(r"[^\u0B00-\u0B7F\s।,;:!?()\-\"'“”‘’/]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[।!?])\s+|\n+")


@dataclass(frozen=True)
class GeneratorConfig:
    fonts_folder: Path
    output_folder: Path
    num_images: int
    image_width: int
    image_height: int
    font_size_min: int
    font_size_max: int
    train_split: float
    augmentation_prob: float
    preview: bool
    preview_count: int
    show_preview: bool
    random_seed: int
    wikipedia_target_sentences: int
    wikipedia_max_pages: int


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate a synthetic Odia OCR dataset.")
    parser.add_argument("--preview", action="store_true", help="Generate only preview samples.")
    parser.add_argument("--show-preview", action="store_true", help="Attempt to open preview images after generation.")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES, help="Total number of images to generate.")
    parser.add_argument("--fonts-folder", default=FONTS_FOLDER, help="Folder containing Odia .ttf fonts.")
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, help="Output dataset directory.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--target-sentences",
        type=int,
        default=WIKIPEDIA_TARGET_SENTENCES,
        help="Target number of unique Odia sentences to scrape.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=WIKIPEDIA_MAX_PAGES,
        help="Maximum Odia Wikipedia pages to crawl.",
    )
    args = parser.parse_args()

    num_images = args.preview and PREVIEW_COUNT or args.num_images
    return GeneratorConfig(
        fonts_folder=Path(args.fonts_folder),
        output_folder=Path(args.output_folder),
        num_images=num_images,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        font_size_min=FONT_SIZE_MIN,
        font_size_max=FONT_SIZE_MAX,
        train_split=TRAIN_SPLIT,
        augmentation_prob=AUGMENTATION_PROB,
        preview=args.preview,
        preview_count=PREVIEW_COUNT,
        show_preview=args.show_preview,
        random_seed=args.seed,
        wikipedia_target_sentences=args.target_sentences,
        wikipedia_max_pages=args.max_pages,
    )


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def require_albumentations() -> Any:
    try:
        import albumentations as album
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "albumentations is required to run this script. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return album


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_sentence(text: str) -> str:
    filtered = ALLOWED_TEXT_PATTERN.sub(" ", text)
    filtered = re.sub(r"\s+", " ", filtered).strip()
    if len(filtered) < 3:
        return ""
    if not ODIA_CHAR_PATTERN.search(filtered):
        return ""
    return filtered


def split_into_sentences(text: str) -> list[str]:
    raw_parts = SENTENCE_SPLIT_PATTERN.split(text)
    sentences: list[str] = []
    for part in raw_parts:
        sentence = clean_sentence(part)
        if sentence:
            sentences.append(sentence)
    return sentences


def scrape_odia_sentences(config: GeneratorConfig, rng: random.Random) -> list[str]:
    wiki = wikipediaapi.Wikipedia(user_agent="odia-dataset-generator/1.0", language="or")
    queue = list(DEFAULT_SEED_TITLES)
    seen_titles: set[str] = set()
    collected: list[str] = []
    unique_sentences: set[str] = set()

    with tqdm(total=config.wikipedia_target_sentences, desc="Scraping Odia text", unit="sentence") as progress:
        while queue and len(seen_titles) < config.wikipedia_max_pages and len(unique_sentences) < config.wikipedia_target_sentences:
            title = queue.pop(0)
            if title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                page = wiki.page(title)
            except Exception as exc:
                logging.warning("Skipping page '%s': %s", title, exc)
                continue
            if not page.exists():
                continue

            page_text = "\n".join(filter(None, [page.summary, page.text]))
            before_count = len(unique_sentences)
            for sentence in split_into_sentences(page_text):
                if sentence in unique_sentences:
                    continue
                unique_sentences.add(sentence)
                collected.append(sentence)
                if len(unique_sentences) >= config.wikipedia_target_sentences:
                    break
            progress.update(len(unique_sentences) - before_count)

            links = list(page.links.keys())
            rng.shuffle(links)
            for link_title in links:
                if link_title not in seen_titles:
                    queue.append(link_title)
            if len(queue) > config.wikipedia_max_pages * 20:
                queue = queue[: config.wikipedia_max_pages * 20]

    if not collected:
        logging.warning("No Odia text could be scraped. Falling back to bundled sample sentences.")
        return FALLBACK_SENTENCES.copy()

    if len(collected) < min(config.num_images, config.wikipedia_target_sentences // 5):
        logging.warning(
            "Only %s unique sentences were collected from Odia Wikipedia. Generation will continue with the available corpus.",
            len(collected),
        )
    return collected


def save_sentences(sentences: Sequence[str], output_path: Path) -> None:
    output_path.write_text("\n".join(sentences), encoding="utf-8")


def font_supports_odia(font_path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), size=max(FONT_SIZE_MIN, 28))
        bbox = font.getbbox(ODIA_SAMPLE_TEXT)
        return bbox is not None and bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0
    except Exception:
        return False


def discover_fonts(fonts_folder: Path) -> list[Path]:
    if not fonts_folder.exists():
        raise FileNotFoundError(f"Fonts folder does not exist: {fonts_folder}")

    supported_fonts: list[Path] = []
    for font_path in sorted(fonts_folder.glob("*.ttf")):
        if font_supports_odia(font_path):
            supported_fonts.append(font_path)
        else:
            logging.warning("Skipping font without usable Odia glyphs: %s", font_path.name)

    if not supported_fonts:
        raise RuntimeError("No Odia-capable .ttf fonts were found in the fonts folder.")
    return supported_fonts


def create_paper_texture(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    base = Image.new("RGB", size, color=(245, 241, 232))
    draw = ImageDraw.Draw(base)

    line_spacing = rng.randint(12, 20)
    line_color = (228, 221, 206)
    for y_pos in range(line_spacing, height, line_spacing):
        draw.line([(0, y_pos), (width, y_pos)], fill=line_color, width=1)

    noise = np.random.normal(loc=0.0, scale=4.0, size=(height, width, 1)).astype(np.int16)
    texture = np.clip(np.array(base, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(texture, mode="RGB")


def fit_font_for_text(text: str, font_path: Path, config: GeneratorConfig, rng: random.Random) -> ImageFont.FreeTypeFont:
    for font_size in range(rng.randint(config.font_size_min, config.font_size_max), config.font_size_min - 1, -1):
        font = ImageFont.truetype(str(font_path), size=font_size)
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= config.image_width - 16 and text_height <= config.image_height - 12:
            return font
    return ImageFont.truetype(str(font_path), size=config.font_size_min)


def render_text_image(text: str, font_path: Path, config: GeneratorConfig, rng: random.Random) -> Image.Image:
    background_choice = rng.random()
    if background_choice < 0.7:
        bg_tone = rng.randint(244, 255)
        image = Image.new("RGB", (config.image_width, config.image_height), color=(bg_tone, bg_tone, bg_tone))
    else:
        image = create_paper_texture((config.image_width, config.image_height), rng)

    font = fit_font_for_text(text, font_path, config, rng)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    max_x = max(8, config.image_width - text_width - 8)
    max_y = max(4, config.image_height - text_height - 4)
    x_pos = rng.randint(6, max_x)
    y_pos = rng.randint(2, max_y)
    ink = rng.randint(5, 45)
    draw.text((x_pos, y_pos), text, fill=(ink, ink, ink), font=font)

    if rng.random() < 0.25:
        offset = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, rng.uniform(-1.0, 1.0), 0, 1, rng.uniform(-1.0, 1.0)),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )
        image = Image.blend(image, offset, alpha=0.15)

    return image


def apply_random_augmentations(image: Image.Image, rng: random.Random) -> Image.Image:
    album = require_albumentations()
    working = np.array(image)

    operations = ["rotate", "blur", "noise", "brightness", "elastic", "texture"]
    rng.shuffle(operations)
    count = rng.randint(2, 3)
    chosen = operations[:count]

    if "texture" in chosen:
        texture = create_paper_texture(image.size, rng)
        working = np.array(Image.blend(Image.fromarray(working), texture, alpha=0.14))

    transform_map = {
        "rotate": album.Rotate(limit=15, border_mode=0, value=(255, 255, 255), p=1.0),
        "blur": album.GaussianBlur(blur_limit=(3, 5), p=1.0),
        "noise": album.GaussNoise(var_limit=(10.0, 35.0), p=1.0),
        "brightness": album.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        "elastic": album.ElasticTransform(alpha=1.0, sigma=8.0, alpha_affine=4.0, border_mode=0, value=(255, 255, 255), p=1.0),
    }

    for name in chosen:
        if name == "texture":
            continue
        transform = transform_map[name]
        working = transform(image=working)["image"]

    pil_image = Image.fromarray(working)
    if rng.random() < 0.35:
        pil_image = pil_image.filter(ImageFilter.SHARPEN)
    return pil_image


def choose_sentences(sentences: Sequence[str], count: int, rng: random.Random) -> list[str]:
    if not sentences:
        raise RuntimeError("No sentences are available for image generation.")
    if count <= len(sentences):
        chosen = list(sentences)
        rng.shuffle(chosen)
        return chosen[:count]
    return [rng.choice(sentences) for _ in range(count)]


def split_sentence_corpus(sentences: Sequence[str], config: GeneratorConfig, rng: random.Random) -> tuple[list[str], list[str]]:
    unique_sentences = list(dict.fromkeys(sentences))
    rng.shuffle(unique_sentences)
    split_index = max(1, int(len(unique_sentences) * config.train_split))
    split_index = min(split_index, len(unique_sentences) - 1) if len(unique_sentences) > 1 else len(unique_sentences)
    train_sentences = unique_sentences[:split_index] or unique_sentences
    test_sentences = unique_sentences[split_index:] if len(unique_sentences) > 1 else unique_sentences
    return train_sentences, test_sentences


def save_image(image: Image.Image, path: Path) -> int:
    image.save(path, quality=JPEG_QUALITY)
    return path.stat().st_size


def generate_split(
    split_name: str,
    split_sentences: Sequence[str],
    image_count: int,
    start_index: int,
    fonts: Sequence[Path],
    config: GeneratorConfig,
    rng: random.Random,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[int], int]:
    rows: list[dict[str, str]] = []
    file_sizes: list[int] = []
    skipped = 0
    chosen_sentences = choose_sentences(split_sentences, image_count, rng)
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(chosen_sentences, desc=f"Generating {split_name}", unit="image")
    for offset, sentence in enumerate(progress, start=start_index):
        file_name = f"img_{offset:05d}{IMAGE_EXTENSION}"
        file_path = split_dir / file_name
        try:
            font_path = rng.choice(list(fonts))
            image = render_text_image(sentence, font_path, config, rng)
            if rng.random() < config.augmentation_prob:
                image = apply_random_augmentations(image, rng)
            file_sizes.append(save_image(image, file_path))
            rows.append({"image_name": f"{split_name}/{file_name}", "text": sentence})
        except Exception as exc:
            skipped += 1
            logging.warning("Skipping image %s for split %s: %s", file_name, split_name, exc)
    return pd.DataFrame(rows), file_sizes, skipped


def write_labels(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8")


def maybe_show_preview(preview_dir: Path) -> None:
    for image_path in sorted(preview_dir.glob(f"*{IMAGE_EXTENSION}")):
        try:
            Image.open(image_path).show(title=image_path.name)
        except Exception as exc:
            logging.warning("Could not display preview image %s: %s", image_path.name, exc)


def preview_dataset(sentences: Sequence[str], fonts: Sequence[Path], config: GeneratorConfig, rng: random.Random) -> None:
    preview_dir = config.output_folder / "preview"
    ensure_clean_dir(preview_dir)
    chosen = choose_sentences(sentences, config.preview_count, rng)
    for index, sentence in enumerate(tqdm(chosen, desc="Generating preview", unit="image"), start=1):
        font_path = rng.choice(list(fonts))
        image = render_text_image(sentence, font_path, config, rng)
        if rng.random() < config.augmentation_prob:
            image = apply_random_augmentations(image, rng)
        file_path = preview_dir / f"preview_{index:02d}{IMAGE_EXTENSION}"
        save_image(image, file_path)

    logging.info("Preview images written to %s", preview_dir)
    if config.show_preview:
        maybe_show_preview(preview_dir)


def report_stats(
    total_generated: int,
    unique_sentences_used: int,
    file_sizes: Sequence[int],
    skipped: int,
    started_at: float,
) -> None:
    elapsed = time.time() - started_at
    average_kb = statistics.mean(file_sizes) / 1024 if file_sizes else 0.0
    logging.info("Total images generated: %s", total_generated)
    logging.info("Unique sentences available: %s", unique_sentences_used)
    logging.info("Skipped images: %s", skipped)
    logging.info("Elapsed time: %.2f seconds", elapsed)
    logging.info("Average image size: %.2f KB", average_kb)


def generate_dataset(sentences: Sequence[str], fonts: Sequence[Path], config: GeneratorConfig, rng: random.Random) -> None:
    ensure_clean_dir(config.output_folder)
    save_sentences(sentences, config.output_folder / "odia_sentences.txt")

    train_sentences, test_sentences = split_sentence_corpus(sentences, config, rng)
    train_count = max(1, math.floor(config.num_images * config.train_split))
    test_count = max(1, config.num_images - train_count)
    started_at = time.time()

    train_df, train_sizes, train_skipped = generate_split(
        split_name="train",
        split_sentences=train_sentences,
        image_count=train_count,
        start_index=1,
        fonts=fonts,
        config=config,
        rng=rng,
        output_dir=config.output_folder,
    )
    test_df, test_sizes, test_skipped = generate_split(
        split_name="test",
        split_sentences=test_sentences,
        image_count=test_count,
        start_index=train_count + 1,
        fonts=fonts,
        config=config,
        rng=rng,
        output_dir=config.output_folder,
    )

    write_labels(train_df, config.output_folder / "train_labels.csv")
    write_labels(test_df, config.output_folder / "test_labels.csv")
    report_stats(
        total_generated=len(train_df) + len(test_df),
        unique_sentences_used=len(set(sentences)),
        file_sizes=train_sizes + test_sizes,
        skipped=train_skipped + test_skipped,
        started_at=started_at,
    )


def main() -> None:
    configure_logging()
    config = parse_args()
    rng = random.Random(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    fonts = discover_fonts(config.fonts_folder)
    sentences = scrape_odia_sentences(config, rng)

    if config.preview:
        ensure_clean_dir(config.output_folder)
        save_sentences(sentences, config.output_folder / "odia_sentences.txt")
        preview_dataset(sentences, fonts, config, rng)
        return

    generate_dataset(sentences, fonts, config, rng)


if __name__ == "__main__":
    main()