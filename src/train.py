"""
src/train.py — Training loop for BioBERT + LoRA/QLoRA on BC5CDR NER.

Pipeline:
  TrainingConfig (or QLoRATrainingConfig)
    → get_datasets()   (data.py)
    → get_model()      (model.py)
    → Trainer.train()
    → saved adapter in cfg.output_dir

Entry points:
    train(cfg)  — run full training from a config object
    main()      — CLI; use --qlora flag to switch to QLoRA
"""

import argparse
import inspect
import logging
import os
from typing import Dict, List

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import QLoRATrainingConfig, TrainingConfig
from src.data import get_datasets, get_label_info
from src.model import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: Dict[int, str]):
    """Return a compute_metrics closure bound to the label mapping.

    Why a closure? Trainer.compute_metrics only accepts (EvalPrediction) → dict,
    with no way to pass extra arguments. Wrapping it in a closure lets us bake in
    id2label without a global variable.

    seqeval operates on WORD-level label sequences, not subword-level. It expects:
        predictions : List[List[str]]  — one label string per word
        references  : List[List[str]]  — same shape

    We get SUBWORD-level tensors from the model (shape: [batch, seq_len, num_labels]).
    Steps to convert:
        1. Argmax over the label dimension → integer token predictions
        2. Zip predictions with gold labels; skip positions where gold == -100
           (those are [CLS]/[SEP]/[PAD] and continuation subwords — not word-level)
        3. Convert surviving integers to strings via id2label
        4. Feed the resulting string lists to seqeval

    Skipping -100 positions is correct for token-level NER because seqeval counts
    entity spans, not individual tokens. A span is correct only if ALL its tokens
    are predicted correctly — hiding subword tokens from the metric matches how
    we hid them from the loss during training.
    """
    def compute_metrics(eval_pred) -> Dict[str, float]:
        raw_logits, raw_labels = eval_pred

        # raw_logits can be a tuple (logits, hidden_states, attentions) when
        # output_hidden_states=True. We always want position 0 (the actual logits).
        if isinstance(raw_logits, tuple):
            raw_logits = raw_logits[0]

        # Argmax → integer class IDs, shape [batch, seq_len]
        predictions = np.argmax(raw_logits, axis=-1)

        true_labels: List[List[str]] = []
        pred_labels: List[List[str]] = []

        for pred_seq, label_seq in zip(predictions, raw_labels):
            true_row: List[str] = []
            pred_row: List[str] = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    # Special / continuation token — skip entirely
                    continue
                true_row.append(id2label[int(label_id)])
                pred_row.append(id2label[int(pred_id)])

            true_labels.append(true_row)
            pred_labels.append(pred_row)

        return {
            # seqeval computes SPAN-level (entity-level) metrics, not token-level.
            # An entity is correct only if its entire span matches — both boundaries
            # and type. This is the standard BC5CDR evaluation protocol.
            "f1":        f1_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall":    recall_score(true_labels, pred_labels),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Weighted loss Trainer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to the token-classification loss.

    Why subclass instead of using label_smoothing or a built-in option?
    HuggingFace Trainer's built-in CrossEntropyLoss is unweighted. The only hook
    for replacing the loss function is overriding compute_loss — there's no
    TrainingArguments field for class weights.

    Weight order must match DataConfig.label_names:
        [O, B-Chemical, I-Chemical, B-Disease, I-Disease]
        [1.0,   5.0,       5.0,       10.0,      10.0  ]

    We pop "labels" from inputs before the forward pass so the model doesn't
    compute its own (unweighted) internal loss — we compute ours instead.
    The returned loss is what Trainer uses for the backward pass.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register as a buffer-like attribute; moved to the correct device in compute_loss.
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")  # [batch, seq_len, num_labels]

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,  # skip [CLS]/[SEP]/[PAD] and subword continuations
        )
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# TrainingArguments builder
# ---------------------------------------------------------------------------

def build_training_args(cfg: TrainingConfig) -> TrainingArguments:
    """Translate our TrainingConfig dataclass into a HuggingFace TrainingArguments.

    We keep all numeric hyperparameters in TrainingConfig (config.py) and only
    wire them through here. That way the config file is the single source of truth
    for any hyperparameter search — you never need to grep train.py for magic numbers.

    Notable settings:
    - label_names=["labels"]: tells Trainer which dataset column holds the labels,
      so it can pass them correctly into the model's forward() call.
    - remove_unused_columns=False: by default Trainer drops any dataset column that
      isn't a model input. Since we already stripped non-tensor columns in data.py,
      this is just defensive. Flip to True and Trainer silently drops "labels" if
      the model's forward() signature doesn't list it — very hard to debug.
    - eval_on_start=False: skip an initial eval pass before any training.
      The base model predicts noise for NER, so the initial eval is uninformative
      and wastes time on Kaggle's per-session GPU quota.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    args_kwargs = {
        "output_dir": cfg.output_dir,

        # Batch sizes
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,

        # Learning rate schedule
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": cfg.num_train_epochs,

        # Eval and checkpointing
        "save_strategy": cfg.save_strategy,
        "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": cfg.load_best_model_at_end,
        "metric_for_best_model": cfg.metric_for_best_model,
        "greater_is_better": cfg.greater_is_better,

        # Precision
        "fp16": cfg.fp16 and torch.cuda.is_available(),

        # Logging
        "logging_steps": cfg.logging_steps,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,

        # Misc
        "seed": cfg.seed,
        "label_names": ["labels"],
        "remove_unused_columns": False,
    }

    # Transformers renamed this argument across versions. Supporting both keeps
    # the project usable locally and in Kaggle's pinned environment.
    signature = inspect.signature(TrainingArguments.__init__)
    strategy_arg = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    args_kwargs[strategy_arg] = cfg.eval_strategy

    if "eval_on_start" in signature.parameters:
        args_kwargs["eval_on_start"] = False

    if cfg.fp16 and not torch.cuda.is_available():
        logger.warning("CUDA is not available; disabling fp16 for this run.")

    return TrainingArguments(**args_kwargs)


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    cfg: TrainingConfig,
    model,
    datasets,
    tokenizer,
    id2label: Dict[int, str],
) -> WeightedTrainer:
    """Assemble the HuggingFace Trainer.

    DataCollatorForTokenClassification:
        Pads sequences within each batch to the longest sequence IN THAT BATCH,
        not to the global max_length (512). This typically cuts padding by 60-70%
        vs static padding, reducing compute and memory per batch.
        It also pads the labels tensor in parallel, using -100 as the pad label
        so those positions are automatically ignored by the loss.

    WeightedTrainer:
        Uses cfg.class_weights to upweight B-Disease / I-Disease tokens in the
        cross-entropy loss. Standard Trainer predicts "O" for most Disease tokens
        because O dominates the token distribution. Weighting forces the loss to
        penalise Disease misses more heavily, improving recall for that class.

    EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience):
        Stop training if eval_f1 has not improved for N consecutive evaluations.
        N=3 (raised from 2) gives Disease entities more time to converge after
        Chemical entities plateau — patience=2 fired too early in prior runs.
        Requires load_best_model_at_end=True (already set in TrainingConfig).
    """
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,         # pad to longest in batch
        label_pad_token_id=-100,
    )

    training_args = build_training_args(cfg)
    class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    return trainer


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def train(cfg: TrainingConfig) -> Trainer:
    """Run the full training pipeline and return the finished Trainer.

    Returns the Trainer so callers can do post-hoc analysis (e.g., access
    trainer.state.log_history for a custom loss plot) without re-running training.

    The adapter is saved to cfg.output_dir automatically by Trainer when
    load_best_model_at_end=True — it saves the best checkpoint, not the last.
    """
    set_seed(cfg.seed)

    logger.info("=== Starting training | output_dir=%s ===", cfg.output_dir)
    logger.info("Config: %s", cfg)

    # --- Data ------------------------------------------------------------------
    datasets, tokenizer = get_datasets(cfg.data)
    _, label2id, id2label = get_label_info(cfg.data)

    logger.info(
        "Dataset sizes — train: %d | validation: %d | test: %d",
        len(datasets["train"]),
        len(datasets["validation"]),
        len(datasets["test"]),
    )

    # --- Model -----------------------------------------------------------------
    model = get_model(cfg, label2id, id2label)

    # --- Train -----------------------------------------------------------------
    trainer = build_trainer(cfg, model, datasets, tokenizer, id2label)

    logger.info("Starting Trainer.train()...")
    train_result = trainer.train()

    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Save the final adapter weights and tokenizer alongside the checkpoint.
    # model.save_pretrained() on a PeftModel saves ONLY the adapter weights
    # (adapter_config.json + adapter_model.safetensors) — NOT the 110M base model.
    # This is the main efficiency win of PEFT: a ~5MB adapter instead of a ~440MB checkpoint.
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    logger.info("Adapter saved to %s", cfg.output_dir)
    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BioBERT on BC5CDR with LoRA or QLoRA")
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use QLoRA (4-bit base model) instead of standard LoRA",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: from config)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated by W&B)",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default=None,
        help='Experiment tracker target, e.g. "wandb" or "none" (default: from config)',
    )
    args = parser.parse_args()

    if args.qlora:
        cfg = QLoRATrainingConfig()
        logger.info("Using QLoRA configuration (4-bit quantization)")
    else:
        cfg = TrainingConfig()
        logger.info("Using standard LoRA configuration")

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.report_to:
        cfg.report_to = args.report_to

    train(cfg)


if __name__ == "__main__":
    main()
