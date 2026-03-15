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
    eval_samples: int  # max validation samples per eval; -1 = all
    val_ratio: float
    auto_resume: bool
    resume_checkpoint: Path | None


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
        help="Max number of validation samples used for evaluation each epoch. -1 = use all validation samples. Default: 500.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train split used as validation (0 < val_ratio < 1). Default: 0.1.",
    )
    parser.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Resume from the latest local checkpoint if available.",
    )
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Start training from scratch even when a local checkpoint exists.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional path to a specific checkpoint .pt file.",
    )
    parser.set_defaults(use_mixed_precision=True, use_8bit_adam=True, auto_resume=True)
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
        val_ratio=args.val_ratio,
        auto_resume=args.auto_resume,
        resume_checkpoint=Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None,
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


def save_metadata(
    output_dir: Path,
    config: TrainingConfig,
    best_cer: float,
    best_epoch: int,
    final_test_cer: float | None = None,
) -> None:
    metadata = {
        "config": asdict(config),
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "final_test_cer": final_test_cer,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def get_latest_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "latest_checkpoint.pt"


def save_checkpoint(
    checkpoint_path: Path,
    model: VisionEncoderDecoderModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_cer: float,
    best_epoch: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: VisionEncoderDecoderModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_completed_epoch = int(checkpoint.get("epoch", 0))
    best_cer = float(checkpoint.get("best_cer", float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    return last_completed_epoch, best_cer, best_epoch


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
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

    train_indices = np.arange(len(train_df))
    rng = np.random.default_rng(config.seed)
    rng.shuffle(train_indices)
    val_count = max(1, int(len(train_df) * config.val_ratio))
    val_indices = train_indices[:val_count]
    fit_indices = train_indices[val_count:]
    if len(fit_indices) == 0:
        raise ValueError("val_ratio is too high; no samples left for training.")

    fit_df = train_df.iloc[fit_indices].reset_index(drop=True)
    val_df = train_df.iloc[val_indices].reset_index(drop=True)

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

    train_dataset = OdiaOCRDataset(fit_df, config.dataset_root)
    eval_df = val_df if config.eval_samples < 0 else val_df.iloc[: config.eval_samples]
    eval_dataset = OdiaOCRDataset(eval_df, config.dataset_root)
    test_dataset = OdiaOCRDataset(test_df, config.dataset_root)
    collate_fn = build_collate_fn(processor, config.max_target_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
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
    print(f"Train samples: {len(fit_df)}, validation samples: {len(val_df)}, test samples: {len(test_df)}")
    print(f"Evaluating on {len(eval_df)} validation samples per epoch (pass --eval-samples -1 for full validation).")

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
    last_completed_epoch = 0
    checkpoint_path = config.resume_checkpoint or get_latest_checkpoint_path(config.output_dir)

    if config.auto_resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        last_completed_epoch, best_cer, best_epoch = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        print(f"Resumed at epoch {last_completed_epoch}. Next epoch: {last_completed_epoch + 1}")
    elif config.resume_checkpoint and not checkpoint_path.exists():
        raise FileNotFoundError(f"Requested checkpoint not found: {checkpoint_path}")

    start_epoch = last_completed_epoch + 1
    end_epoch = last_completed_epoch + config.epochs
    print(f"Training this run for epochs {start_epoch} to {end_epoch}.")

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = 0.0

        accum_steps = max(1, config.grad_accum_steps)
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}", unit="batch")):
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
            dataloader=eval_loader,
            device=device,
            max_target_length=config.max_target_length,
            num_beams=config.num_beams,
            use_mixed_precision=config.use_mixed_precision,
        )
        print(f"Epoch {epoch}: train_loss={average_loss:.4f} val_cer={cer_score:.4f}")

        if cer_score < best_cer:
            best_cer = cer_score
            best_epoch = epoch
            model.save_pretrained(config.output_dir)
            processor.save_pretrained(config.output_dir)
        save_checkpoint(
            checkpoint_path=get_latest_checkpoint_path(config.output_dir),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_cer=best_cer,
            best_epoch=best_epoch,
        )
        save_metadata(config.output_dir, config, best_cer, best_epoch)

    final_test_cer = evaluate(
        model=model,
        processor=processor,
        dataloader=test_loader,
        device=device,
        max_target_length=config.max_target_length,
        num_beams=config.num_beams,
        use_mixed_precision=config.use_mixed_precision,
    )
    save_metadata(config.output_dir, config, best_cer, best_epoch, final_test_cer=final_test_cer)
    print(f"Final test CER (one-time evaluation): {final_test_cer:.4f}")

    print(f"Best model saved to {config.output_dir} with CER={best_cer:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()