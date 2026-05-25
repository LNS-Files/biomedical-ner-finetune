"""
Central configuration for all hyperparameters, paths, and experiment settings.

Design rule: every number that affects training lives here. If you need to explain
a choice in an interview, the comment next to it is your answer.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # HuggingFace dataset identifier for BC5CDR via the BigBIO schema
    # bigbio/bc5cdr exposes a standardized NER schema; the "bc5cdr_bigbio_kb" config
    # gives us passage-level examples with entity spans — easier to align than raw BioC XML
    dataset_name: str = "bigbio/bc5cdr"
    dataset_config: str = "bc5cdr_bigbio_kb"

    # BioBERT's tokenizer — must match the base model exactly
    tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.2"

    # WordPiece splits words into subword tokens. We truncate/pad all sequences to
    # 512 tokens (BioBERT's max positional embedding limit). BC5CDR abstracts are
    # typically 150-300 tokens, so very few sequences actually hit this ceiling.
    max_length: int = 512

    # Label scheme used in the processed dataset. BIO tagging:
    #   B- = Beginning of an entity span
    #   I- = Inside (continuation) of a span
    #   O  = Outside any entity
    # The order here defines the integer IDs used in the label tensor.
    label_names: List[str] = field(default_factory=lambda: [
        "O",
        "B-Chemical",
        "I-Chemical",
        "B-Disease",
        "I-Disease",
    ])

    # Fraction of the training set used as a dev/validation set.
    # BC5CDR has official train/dev/test splits — we use those directly and
    # this field is a fallback only if the split isn't pre-defined.
    validation_split: float = 0.1

    # Reproducibility seed for any random splits or data shuffling
    seed: int = 42


@dataclass
class LoraConfig:
    # LoRA decomposes weight updates as W' = W + (A @ B) * (alpha / rank).
    # rank=32: raised from 16 to give the adapter more capacity to capture Disease
    # entity patterns, which are contextually richer than Chemical names.
    # Lower rank (4-8) = fewer parameters, higher regularization, may underfit.
    # Higher rank (32-64) = more expressive but risks overfitting on small datasets.
    r: int = 32

    # alpha controls the effective learning rate of the LoRA update.
    # The weight update is scaled by alpha/rank. Setting alpha = 2*rank (here: 64)
    # keeps the update scale constant regardless of rank choice — this means you can
    # change rank without retuning your learning rate. It's the standard default.
    lora_alpha: int = 64

    # Dropout on the LoRA layers. 0.1 is a light regularizer; BC5CDR training set
    # is ~4500 sentences, small enough that some dropout helps generalization.
    lora_dropout: float = 0.1

    # Which weight matrices to apply LoRA to.
    # BERT's attention block has 4 matrices: query (q), key (k), value (v),
    # and output projection (o). Adding "key" to the original q+v recommendation
    # gives the adapter more control over attention routing, which helps with
    # contextual Disease entities whose boundaries depend on surrounding words.
    target_modules: List[str] = field(default_factory=lambda: ["query", "key", "value"])

    # Bias handling: "none" means we don't train any bias terms in LoRA layers.
    # "none" is standard — bias parameters are a tiny fraction of total params and
    # training them adds negligible benefit while complicating adapter serialization.
    bias: str = "none"

    # Task type tells PEFT how to wire the adapter output. TOKEN_CLS routes LoRA
    # through BioBERT's hidden states before the linear classification head.
    task_type: str = "TOKEN_CLS"


@dataclass
class QLoRAConfig(LoraConfig):
    """QLoRA variant: same LoRA settings, adds 4-bit quantization of the base model.

    QLoRA loads the frozen base model in NF4 (Normal Float 4) quantization.
    The LoRA adapters themselves remain in full precision (bfloat16).
    Net effect: ~4x memory reduction on the base model weights.
    On Kaggle P100 (16GB), this frees enough VRAM to increase per-device batch size
    or add more target modules without hitting the 16GB ceiling.
    """
    # Whether to load the base model in 4-bit
    load_in_4bit: bool = True

    # NF4 (NormalFloat4) is the quantization data type introduced in the QLoRA paper.
    # It outperforms INT4/FP4 on transformer weights because those weights are
    # approximately normally distributed — NF4 spaces quantization levels to match
    # that distribution, minimizing quantization error.
    bnb_4bit_quant_type: str = "nf4"

    # The compute dtype: operations happen in bfloat16 even though weights are stored
    # in 4-bit. bfloat16 has the same exponent range as float32 (prevents overflow)
    # but half the mantissa bits. P100 doesn't have native bf16 hardware support
    # but PyTorch handles the emulation — it's still faster than float32 on P100.
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Double quantization: quantize the quantization constants themselves, saving
    # an additional ~0.4 bits/parameter. Negligible compute overhead, free memory.
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    # Output directory for checkpoints and final adapter weights
    output_dir: str = "results/lora-biobert-bc5cdr"

    # Base model — BioBERT-base trained on PubMed + PMC text.
    # Using the cased variant because chemical/disease names are case-sensitive
    # (e.g., "Aspirin" vs "aspirin" appear differently in abstracts).
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"

    # --- Batch size and gradient accumulation ---
    # P100 has 16GB VRAM. BioBERT-base is ~110M params.
    # With LoRA, only ~1-2M params are trainable but the full model stays in memory.
    # per_device_train_batch_size=16 uses ~10GB, leaving headroom for gradients.
    # Effective batch size = 16 * 2 = 32, which is the standard NER fine-tuning size.
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32   # Eval is forward-pass only, can be larger
    gradient_accumulation_steps: int = 2

    # --- Learning rate schedule ---
    # 2e-4 is higher than typical full fine-tuning (2e-5) because LoRA parameters
    # are freshly initialized — they need a larger learning rate to learn quickly
    # while the frozen base model weights provide stable representations.
    learning_rate: float = 2e-4

    # Warmup for 10% of total training steps. During warmup, LR ramps linearly from
    # 0 to learning_rate. This prevents early catastrophic updates when the LoRA
    # matrices are randomly initialized and gradients are noisy.
    warmup_ratio: float = 0.1

    # Weight decay applied via AdamW. L2 regularization on weights (not biases).
    # 0.01 is the standard default for transformers — light enough not to impede
    # learning, strong enough to discourage large weight magnitudes.
    weight_decay: float = 0.01

    # Number of training epochs. Raised from 5 → 10 because Disease entities require
    # more gradient steps than Chemical entities (which are morphologically distinctive
    # and learned quickly). Early stopping still caps the run if the model plateaus.
    num_train_epochs: int = 10

    # How many consecutive non-improving epochs before early stopping fires.
    # Raised from 2 → 3: with 10 max epochs and two entity types that may converge
    # at different rates, patience=2 was too aggressive and killed training before
    # Disease recall had a chance to climb.
    early_stopping_patience: int = 3

    # Evaluate and checkpoint once per epoch. With ~4500 training samples and
    # batch size 16, one epoch is ~280 steps — frequent enough for early stopping.
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"

    # Keep only the top 2 checkpoints on disk (saves Kaggle storage quota).
    save_total_limit: int = 2

    # Load the checkpoint with the best eval metric when training finishes.
    # Without this, you'd get the *last* checkpoint, not the *best* one.
    load_best_model_at_end: bool = True

    # F1 (not loss) is the metric we care about for NER. Loss can decrease while
    # span-level F1 stagnates if the model learns label smoothing artifacts.
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

    # Per-class loss weights for CrossEntropyLoss, ordered to match DataConfig.label_names:
    #   index 0 → O           weight 1.0  (dominant class; no amplification needed)
    #   index 1 → B-Chemical  weight 5.0  (Chemical recall ~72%; moderate boost)
    #   index 2 → I-Chemical  weight 5.0
    #   index 3 → B-Disease   weight 10.0 (Disease recall was only 15%; strong boost)
    #   index 4 → I-Disease   weight 10.0
    # Rule: weight ∝ 1 / class_frequency at the token level, then scaled so "O" = 1.
    # If Disease recall is still low after retraining, raise 10.0 → 15.0.
    class_weights: List[float] = field(default_factory=lambda: [1.0, 5.0, 5.0, 10.0, 10.0])

    # Mixed precision: fp16 on P100. P100 doesn't support bfloat16 natively.
    # fp16=True halves memory for activations and speeds up compute ~1.5x.
    fp16: bool = True

    # --- Logging ---
    logging_steps: int = 50
    report_to: str = "wandb"          # Use CLI --report-to none to disable W&B
    run_name: Optional[str] = None    # W&B run name; None lets W&B auto-generate

    # Reproducibility
    seed: int = 42

    # Composed sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class QLoRATrainingConfig(TrainingConfig):
    """Drop-in replacement for LoRA training. Only differences: output dir and lora config."""
    output_dir: str = "results/qlora-biobert-bc5cdr"
    lora: QLoRAConfig = field(default_factory=QLoRAConfig)
    # fp16 must be False when using bitsandbytes 4-bit; the compute dtype is set in
    # QLoRAConfig.bnb_4bit_compute_dtype instead.
    fp16: bool = False
