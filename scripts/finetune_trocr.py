from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from jiwer import cer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    import bitsandbytes as bnb

    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "odia_dataset"
DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "saved_model"


@dataclass
class TrainingConfig:
    model_name: str
    dataset_root: Path
    output_dir: Path
    batch_size: int
    learning_rate: float
    epochs: int
    max_target_length: int
    num_beams: int
    num_workers: int
    seed: int
    device: str
    use_mixed_precision: bool
    use_8bit_adam: bool
    grad_accum_steps: int
    eval_samples: int  # max test samples per eval; -1 = all


class OdiaOCRDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, dataset_root: Path) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        image_path = self.dataset_root / str(row["image_name"])
        text = str(row["text"])
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "text": text,
            "image_path": str(image_path),
        }


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on the Odia OCR dataset.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam width for evaluation decoding. Default 1 (greedy) is fastest.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Training device. Use 'cuda' or 'cpu'.")
    parser.add_argument(
        "--use-mixed-precision",
        dest="use_mixed_precision",
        action="store_true",
        help="Enable mixed precision on CUDA to reduce VRAM usage.",
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="use_mixed_precision",
        action="store_false",
        help="Disable mixed precision even when CUDA is available.",
    )
    parser.add_argument(
        "--use-8bit-adam",
        dest="use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam (bitsandbytes) to cut optimizer state VRAM by ~75%%.",
    )
    parser.add_argument(
        "--no-8bit-adam",
        dest="use_8bit_adam",
        action="store_false",
        help="Disable 8-bit Adam and use standard AdamW.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps (simulates larger batch size). Default: 8.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Max number of test samples used for evaluation each epoch. -1 = use all 2000. Default: 500.",
    )
    parser.set_defaults(use_mixed_precision=True, use_8bit_adam=True)
    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        dataset_root=Path(args.dataset_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        use_mixed_precision=args.use_mixed_precision,
        use_8bit_adam=args.use_8bit_adam,
        grad_accum_steps=args.grad_accum_steps,
        eval_samples=args.eval_samples,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(dataset_root: Path, split_name: str) -> pd.DataFrame:
    csv_path = dataset_root / f"{split_name}_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels file: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    required_columns = {"image_name", "text"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing_columns)}")
    return dataframe


def build_collate_fn(processor: TrOCRProcessor, max_target_length: int):
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        tokenized = processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "texts": texts,
            "image_paths": [item["image_path"] for item in batch],
        }

    return collate_fn


def evaluate(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    dataloader: DataLoader,
    device: torch.device,
    max_target_length: int,
    num_beams: int,
    use_mixed_precision: bool,
) -> float:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    use_amp = use_mixed_precision and device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, unit="batch"):
            pixel_values = batch["pixel_values"].to(device)
            if use_amp:
                pixel_values = pixel_values.half()
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=max_target_length,
                num_beams=num_beams,
            )
            decoded_predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(prediction.strip() for prediction in decoded_predictions)
            references.extend(reference.strip() for reference in batch["texts"])

    return cer(references, predictions)


def save_metadata(output_dir: Path, config: TrainingConfig, best_cer: float, best_epoch: int) -> None:
    metadata = {
        "config": asdict(config),
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    # Reduce CUDA memory fragmentation (recommended by PyTorch for low-VRAM GPUs)
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Re-run with --device cpu or install CUDA-enabled PyTorch.")

    device = torch.device(config.device)
    use_amp = config.use_mixed_precision and device.type == "cuda"
    train_df = load_split(config.dataset_root, "train")
    test_df = load_split(config.dataset_root, "test")

    processor = TrOCRProcessor.from_pretrained(config.model_name, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = config.max_target_length
    model.to(device)
    if use_amp:
        model.half()  # Convert weights to FP16: saves ~650 MB vs FP32
        # Recompute activations during backward instead of storing them; uses ~30% more compute
        # but reduces peak VRAM by 40-60% during the backward pass.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Model weights converted to FP16 and gradient checkpointing enabled.")

    train_dataset = OdiaOCRDataset(train_df, config.dataset_root)
    eval_df = test_df if config.eval_samples < 0 else test_df.iloc[: config.eval_samples]
    test_dataset = OdiaOCRDataset(eval_df, config.dataset_root)
    collate_fn = build_collate_fn(processor, config.max_target_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    print(f"Evaluating on {len(eval_df)} test samples per epoch (pass --eval-samples -1 for all 2000).")

    if config.use_8bit_adam and _BNB_AVAILABLE and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
        print("Using 8-bit AdamW (bitsandbytes) — optimizer state VRAM reduced ~75%.")
    else:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        if config.use_8bit_adam and not _BNB_AVAILABLE:
            print("Warning: bitsandbytes not available, falling back to standard AdamW.")
    # GradScaler is NOT used: model weights are already FP16 (via model.half()),
    # so gradients are FP16 too. GradScaler expects FP32 grads — combining them raises an error.
    best_cer = float("inf")
    best_epoch = 0
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        accum_steps = max(1, config.grad_accum_steps)
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", unit="batch")):
            pixel_values = batch["pixel_values"].to(device)
            if use_amp:
                pixel_values = pixel_values.half()
            labels = batch["labels"].to(device)

            with nullcontext():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / accum_steps

            loss.backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum_steps

        average_loss = running_loss / max(1, len(train_loader))
        cer_score = evaluate(
            model=model,
            processor=processor,
            dataloader=test_loader,
            device=device,
            max_target_length=config.max_target_length,
            num_beams=config.num_beams,
            use_mixed_precision=config.use_mixed_precision,
        )
        print(f"Epoch {epoch}: train_loss={average_loss:.4f} cer={cer_score:.4f}")

        if cer_score < best_cer:
            best_cer = cer_score
            best_epoch = epoch
            model.save_pretrained(config.output_dir)
            processor.save_pretrained(config.output_dir)
            save_metadata(config.output_dir, config, best_cer, best_epoch)

    print(f"Best model saved to {config.output_dir} with CER={best_cer:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()