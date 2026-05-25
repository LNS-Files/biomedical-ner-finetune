"""
src/evaluate.py — Standalone evaluation of a saved LoRA adapter on the BC5CDR test set.

Usage:
    python -m src.evaluate --no-adapter
    python -m src.evaluate --adapter-path results/lora-biobert-bc5cdr
    python -m src.evaluate --adapter-path results/qlora-biobert-bc5cdr --qlora

Pipeline:
    adapter_path + base model
      → PeftModel (frozen base + loaded adapter weights)  [or base model only for baseline]
      → Trainer.predict() on test split
      → seqeval span-level F1 + per-entity classification report

Separating evaluation from training lets you re-evaluate any saved checkpoint
without re-running training, and makes the eval logic independently testable.
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from src.config import QLoRAConfig, QLoRATrainingConfig, TrainingConfig
from src.data import get_datasets, get_label_info
from src.model import _get_bnb_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_adapter_model(
    adapter_path: Optional[str],
    cfg: TrainingConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> PreTrainedModel:
    """Load the frozen base model and optionally attach a saved LoRA adapter.

    If adapter_path is None, returns the raw base model with no adapter — this
    is the baseline row in the comparison table (BioBERT zero-shot on NER).

    When adapter_path is provided, two-step process (mirrors how PEFT saves adapters):

    Step 1 — Load base model
        We reload the same BioBERT checkpoint used during training. The base model
        weights are identical to training time — the adapter never modifies them.
        For QLoRA we reload in 4-bit to match the memory layout at inference.

    Step 2 — Load adapter via PeftModel.from_pretrained()
        This reads adapter_config.json (the LoRA hyperparameters) and
        adapter_model.safetensors (the trained A/B matrices + classifier head)
        from adapter_path and attaches them to the base model.
        The result is W' = W_frozen + B@A, exactly as during training.

    Why not load the full merged model? Merging (model.merge_and_unload()) creates
    a standard HuggingFace model at full fp32 size (~440MB). Loading the adapter
    separately keeps VRAM usage minimal — useful when evaluating on the same GPU
    used for training.
    """
    load_kwargs: dict = {
        "num_labels": len(label2id),
        "id2label": id2label,
        "label2id": label2id,
        "ignore_mismatched_sizes": True,
    }

    if isinstance(cfg.lora, QLoRAConfig):
        load_kwargs["quantization_config"] = _get_bnb_config(cfg.lora)
        load_kwargs["device_map"] = "auto"
        logger.info("Loading base model in 4-bit NF4 (QLoRA eval mode)")
    else:
        logger.info("Loading base model in full precision")

    base_model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, **load_kwargs)

    if adapter_path is None:
        logger.info("No adapter — returning base model as-is (baseline eval)")
        base_model.eval()
        return base_model

    # Attach the adapter. is_trainable=False puts the model in eval-only mode —
    # all parameters are frozen and no gradient graph is built. This is equivalent
    # to model.eval() + torch.no_grad() but also prevents accidental training.
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()

    logger.info("Adapter loaded from %s", adapter_path)
    return model


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def decode_predictions(
    raw_logits,
    raw_labels: np.ndarray,
    id2label: Dict[int, str],
) -> tuple[List[List[str]], List[List[str]]]:
    """Convert raw model output into seqeval-compatible string sequences.

    raw_logits: shape [n_examples, seq_len, num_labels]  — unnormalized scores
    raw_labels: shape [n_examples, seq_len]              — integer gold labels

    Returns:
        true_labels : List[List[str]]  — gold label strings, -100 positions removed
        pred_labels : List[List[str]]  — predicted label strings, aligned to true_labels

    Skipping -100 positions aligns with how we trained: -100 marks special tokens
    and subword continuations that are not word-level predictions. seqeval expects
    exactly word-level sequences — feeding it subword sequences inflates precision
    by counting trivially-correct "O" predictions for [PAD] tokens.
    """
    if isinstance(raw_logits, tuple):
        raw_logits = raw_logits[0]

    predictions = np.argmax(raw_logits, axis=-1)  # [n, seq_len]

    true_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []

    for pred_seq, label_seq in zip(predictions, raw_labels):
        true_row: List[str] = []
        pred_row: List[str] = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_row.append(id2label[int(label_id)])
            pred_row.append(id2label[int(pred_id)])

        true_labels.append(true_row)
        pred_labels.append(pred_row)

    return true_labels, pred_labels


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(cfg: TrainingConfig, adapter_path: Optional[str] = None) -> Dict[str, float]:
    """Run evaluation on the BC5CDR test split and print a classification report.

    adapter_path=None runs the unmodified base model (baseline row).
    adapter_path=<dir> loads and applies the saved LoRA/QLoRA adapter.

    Uses Trainer.predict() rather than a manual inference loop for two reasons:
        1. Batching is handled automatically (per_device_eval_batch_size from cfg).
        2. DataCollatorForTokenClassification handles within-batch padding, keeping
           memory usage bounded regardless of sequence length variance in the test set.

    Returns the scalar metrics dict for programmatic use (e.g., hyperparameter search).
    """
    run_label = "baseline" if adapter_path is None else os.path.basename(adapter_path.rstrip("/\\"))

    # ---- Data ----------------------------------------------------------------
    datasets, tokenizer = get_datasets(cfg.data)
    _, label2id, id2label = get_label_info(cfg.data)

    test_dataset = datasets["test"]
    logger.info("Test set size: %d passages", len(test_dataset))

    # ---- Model ---------------------------------------------------------------
    model = load_adapter_model(adapter_path, cfg, label2id, id2label)

    # ---- Predict -------------------------------------------------------------
    # We create a minimal Trainer just to get batched prediction — we don't need
    # any training arguments, so we use a temp dir and disable all logging/saving.
    # output_dir is run_label-scoped so parallel runs don't overwrite each other's
    # predict_results.json files.
    eval_args = TrainingArguments(
        output_dir=f"tmp_eval_{run_label}",
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        report_to="none",
        label_names=["labels"],
        remove_unused_columns=False,
    )

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    predictor = Trainer(
        model=model,
        args=eval_args,
        data_collator=collator,
    )

    logger.info("Running Trainer.predict() on test set...")
    predictions_output = predictor.predict(test_dataset)
    # PredictionOutput has: predictions, label_ids, metrics
    raw_logits = predictions_output.predictions
    raw_labels = predictions_output.label_ids

    # ---- Decode and score ----------------------------------------------------
    true_labels, pred_labels = decode_predictions(raw_logits, raw_labels, id2label)

    overall_f1        = f1_score(true_labels, pred_labels)
    overall_precision = precision_score(true_labels, pred_labels)
    overall_recall    = recall_score(true_labels, pred_labels)

    # classification_report gives per-entity-type breakdown.
    # output_dict=False returns a formatted string for printing;
    # the seqeval report shows precision / recall / F1 / support per type.
    report = classification_report(true_labels, pred_labels, digits=4)

    print("\n" + "=" * 60)
    print(f"BC5CDR TEST SET — Entity-level Evaluation  [{run_label}]")
    print("=" * 60)
    print(report)
    print(f"Overall  F1        : {overall_f1:.4f}")
    print(f"Overall  Precision : {overall_precision:.4f}")
    print(f"Overall  Recall    : {overall_recall:.4f}")
    print("=" * 60)

    metrics = {
        "test_f1":        overall_f1,
        "test_precision": overall_precision,
        "test_recall":    overall_recall,
    }

    logger.info("Evaluation complete. Metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BioBERT (baseline) or a LoRA/QLoRA adapter on the BC5CDR test set"
    )

    # --adapter-path and --no-adapter are mutually exclusive; one is required.
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to the saved adapter directory (contains adapter_config.json)",
    )
    source_group.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the raw base BioBERT model without any adapter (baseline)",
    )

    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Load base model in 4-bit (use if the adapter was trained with QLoRA)",
    )
    args = parser.parse_args()

    if args.qlora:
        cfg = QLoRATrainingConfig()
        logger.info("Using QLoRA config for base model loading")
    else:
        cfg = TrainingConfig()
        logger.info("Using standard LoRA config for base model loading")

    adapter_path = None if args.no_adapter else args.adapter_path
    evaluate(cfg, adapter_path)


if __name__ == "__main__":
    main()
