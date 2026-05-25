"""
src/model.py — BioBERT loading with LoRA / QLoRA adapter setup.

Entry point: get_model(cfg, label2id, id2label) → PeftModel ready for training.

LoRA math reminder:
  A frozen weight matrix W (shape d×d) gets a parallel low-rank update:
      W' = W + (B @ A) * (alpha / rank)
  where A is (rank×d) and B is (d×rank), both randomly initialized.
  Only A and B are trained — W never changes.
  Parameter count per matrix: 2 * d * rank
  For BioBERT (d=768, rank=16): 2 * 768 * 16 = 24,576 vs 768*768 = 589,824 (24× fewer)
"""

import logging
from typing import Dict

import torch
from transformers import AutoModelForTokenClassification, BitsAndBytesConfig, PreTrainedModel
from peft import (
    LoraConfig as PeftLoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft import PeftModel

from src.config import LoraConfig, QLoRAConfig, TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_trainable_parameters(model: PreTrainedModel) -> None:
    """Report trainable vs total parameters — the LoRA efficiency story in one line."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)  ←  only these update during training"
    )


# ---------------------------------------------------------------------------
# Internal helpers (prefixed _ — not part of the public API)
# ---------------------------------------------------------------------------

def _get_bnb_config(lora_cfg: QLoRAConfig) -> BitsAndBytesConfig:
    """Build the bitsandbytes 4-bit quantization config for QLoRA.

    NF4 (NormalFloat4) is the quantization format from the QLoRA paper (Dettmers 2023).
    Transformer weight matrices are approximately normally distributed, so NF4 spaces
    its 16 quantization levels to match that distribution — minimising quantization
    error compared to uniform INT4.

    Double quantization quantizes the quantization constants themselves, saving
    roughly 0.4 bits/parameter with negligible compute overhead.
    """
    # The compute dtype is stored as a string in our config ("bfloat16") so we
    # resolve it to an actual torch dtype here.
    compute_dtype = getattr(torch, lora_cfg.bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=lora_cfg.load_in_4bit,
        bnb_4bit_quant_type=lora_cfg.bnb_4bit_quant_type,      # "nf4"
        bnb_4bit_compute_dtype=compute_dtype,                    # torch.bfloat16
        bnb_4bit_use_double_quant=lora_cfg.bnb_4bit_use_double_quant,
    )


def _load_base_model(
    cfg: TrainingConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> PreTrainedModel:
    """Load BioBERT for token classification, with optional 4-bit quantization.

    ignore_mismatched_sizes=True is required because the pretrained BioBERT
    checkpoint was not trained for token classification — its classifier head
    (if any) has a different output dimension than our 5-label NER task.
    HuggingFace would raise an error without this flag; with it, the old head
    is discarded and a fresh randomly-initialized head is attached.

    For QLoRA: device_map="auto" is required by bitsandbytes. It maps model
    layers to available devices. On a single-GPU Kaggle notebook, everything
    goes to cuda:0. Without it, bitsandbytes doesn't know where to place the
    quantized tensors and raises a device placement error.
    """
    load_kwargs = {
        "num_labels": len(label2id),
        "id2label": id2label,
        "label2id": label2id,
        "ignore_mismatched_sizes": True,
    }

    if isinstance(cfg.lora, QLoRAConfig):
        load_kwargs["quantization_config"] = _get_bnb_config(cfg.lora)
        load_kwargs["device_map"] = "auto"
        logger.info("Loading BioBERT in 4-bit NF4 quantization (QLoRA mode)")
    else:
        logger.info("Loading BioBERT in full precision (LoRA mode)")

    model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, **load_kwargs)

    if isinstance(cfg.lora, QLoRAConfig):
        # prepare_model_for_kbit_training does three things:
        #   1. Casts all LayerNorm weights to float32 — prevents NaN gradients
        #      when fp32 norms feed into quantized fp16/bf16 attention layers.
        #   2. Enables gradient checkpointing — recomputes activations on the
        #      backward pass instead of storing them, trading compute for VRAM.
        #   3. Marks the model as ready for gradient computation despite being
        #      in a quantized (non-differentiable) storage format.
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for k-bit training")

    return model


def _apply_lora(model: PreTrainedModel, lora_cfg: LoraConfig) -> PeftModel:
    """Wrap the base model with LoRA adapters via PEFT.

    Two distinct PEFT parameters are easy to confuse:

    target_modules = ["query", "key", "value"]
        Which weight matrices get LoRA adapters. These matrices are split into
        W = W_frozen + B@A. Only A and B are trainable. W_frozen never moves.

    modules_to_save = ["classifier"]
        Which modules are fully unfrozen and saved with the adapter.
        The classifier head does NOT exist in the pretrained checkpoint —
        it's randomly initialized when we call from_pretrained(num_labels=5).
        LoRA can only adapt EXISTING weights; it can't learn from scratch.
        So we must fully train the classifier head, not LoRA-adapt it.
        modules_to_save unfreezes it completely and bundles it into the
        saved adapter so it's restored alongside the LoRA weights at inference.

    Without modules_to_save=["classifier"]:
        The head stays frozen (all-zeros output after initialization),
        the model predicts nothing meaningful, loss never decreases.
        This is a common silent failure in encoder NER + PEFT setups.
    """
    peft_config = PeftLoraConfig(
        task_type=TaskType.TOKEN_CLS,         # encoder token classification
        r=lora_cfg.r,                          # low-rank dimension
        lora_alpha=lora_cfg.lora_alpha,        # scale factor, usually 2 * rank
        lora_dropout=lora_cfg.lora_dropout,    # dropout on adapter layers (0.1)
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,                    # "none" — don't train bias terms
        modules_to_save=["classifier"],        # fully train + save the NER head
    )

    peft_model = get_peft_model(model, peft_config)
    return peft_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(
    cfg: TrainingConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> PeftModel:
    """Load BioBERT, apply LoRA or QLoRA, and report trainable parameter count.

    To switch between LoRA and QLoRA, pass a different cfg:
        LoRA:  get_model(TrainingConfig(), ...)
        QLoRA: get_model(QLoRATrainingConfig(), ...)

    No other code changes. The isinstance checks inside handle the rest.
    """
    base_model = _load_base_model(cfg, label2id, id2label)
    model = _apply_lora(base_model, cfg.lora)
    print_trainable_parameters(model)
    return model
