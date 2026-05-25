"""
src/data.py — BC5CDR dataset loading and token-label alignment.

Pipeline:
  raw bigbio KB documents          (document-level, absolute char offsets)
    → passage-level word/tag lists  (process_document)
    → HuggingFace Dataset           (build_passage_dataset)
    → tokenized + aligned tensors   (tokenize_and_align_labels)
    → DatasetDict                   (get_datasets)
"""

import re
import logging
from functools import partial
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def get_label_info(
    cfg: DataConfig,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Derive label2id and id2label from the config's label_names list.

    Keeping this as the single source of truth means the model head, the
    training loop, and the evaluator all use the same integer assignments.
    """
    label2id = {label: idx for idx, label in enumerate(cfg.label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return cfg.label_names, label2id, id2label


def get_tokenizer(cfg: DataConfig) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    return tokenizer


# ---------------------------------------------------------------------------
# Stage 1: bigbio KB document → passage-level word/tag records
# ---------------------------------------------------------------------------

def process_document(example: dict, label2id: Dict[str, int]) -> List[Dict]:
    """Convert one bigbio KB document into passage-level word/BIO-tag records.

    bigbio KB structure we care about:
      passages: [{text: [str], offsets: [[abs_start, abs_end]]}, ...]
      entities: [{type: str, offsets: [[abs_start, abs_end]]}, ...]

    Entity offsets are ABSOLUTE character positions in the full document.
    Passage offsets tell us where each passage sits in that document.
    We process each passage independently, converting entity offsets to
    passage-relative positions.
    """
    passage_records = []

    for passage in example["passages"]:
        # text is always a list; BC5CDR passages contain exactly one string.
        text: str = passage["text"][0]
        passage_start: int = passage["offsets"][0][0]
        passage_end: int = passage["offsets"][0][1]

        # ----------------------------------------------------------------
        # Step A: Build a character-level BIO label array for this passage.
        #
        # We start with every character labeled 'O', then paint entity spans
        # on top. The first character of a span becomes B-TYPE; the rest I-TYPE.
        #
        # Why character-level first? The bigbio dataset gives us char offsets,
        # not word offsets. Working at the char level lets us handle entities
        # that span punctuation or are embedded mid-word without extra logic.
        # ----------------------------------------------------------------
        char_labels: List[str] = ["O"] * len(text)

        for entity in example["entities"]:
            entity_type: str = entity["type"]  # "Chemical" or "Disease"

            for abs_start, abs_end in entity["offsets"]:
                # Convert absolute document offset → passage-relative offset
                rel_start = abs_start - passage_start
                rel_end = abs_end - passage_start

                # Reject spans outside this passage's bounds
                if rel_start < 0 or rel_end > len(text) or rel_start >= rel_end:
                    continue

                # Paint BIO labels onto the character array
                char_labels[rel_start] = f"B-{entity_type}"
                for char_idx in range(rel_start + 1, rel_end):
                    char_labels[char_idx] = f"I-{entity_type}"

        # ----------------------------------------------------------------
        # Step B: Split text into whitespace-delimited word tokens.
        #
        # We use \S+ (any non-whitespace run) rather than splitting on every
        # punctuation boundary. This preserves hyphenated chemical names like
        # "5-fluorouracil" or "beta-blocker" as single word units.
        # BioBERT's WordPiece tokenizer will further split them internally,
        # but we want the word unit to correspond to the entity span boundary.
        #
        # Word label = label of its FIRST character.
        # Why? Entity spans always start at a word boundary in BC5CDR, so the
        # first character of a word is either 'O', 'B-TYPE' (start of entity),
        # or 'I-TYPE' (continuation inside a multi-word entity). That's exactly
        # what we want for BIO tagging.
        # ----------------------------------------------------------------
        words: List[str] = []
        ner_tags: List[str] = []

        for match in re.finditer(r"\S+", text):
            words.append(match.group())
            ner_tags.append(char_labels[match.start()])

        if words:
            passage_records.append({"words": words, "ner_tags": ner_tags})

    return passage_records


def build_passage_dataset(raw_split, label2id: Dict[str, int]) -> Dataset:
    """Expand document-level examples into passage-level examples.

    We cannot use HuggingFace's .map() here because .map() is 1-to-1:
    one input example → one output example. But each BC5CDR document
    contains two passages (title + abstract), so the expansion is 1-to-2.
    Instead, we iterate, flatten, and reconstruct a new Dataset from a list.
    """
    records: List[Dict] = []
    for example in raw_split:
        records.extend(process_document(example, label2id))

    logger.info(
        "Expanded %d documents → %d passage-level examples",
        len(raw_split),
        len(records),
    )
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Stage 2: word/tag records → tokenized tensors with aligned labels
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(
    batch: dict,
    tokenizer: PreTrainedTokenizerFast,
    label2id: Dict[str, int],
    max_length: int,
) -> dict:
    """Apply WordPiece tokenization and align BIO labels to subword tokens.

    THE WORDPIECE ALIGNMENT PROBLEM IN DETAIL
    -----------------------------------------
    Input (word level):
        words  = ["Naloxone",    "reverses", "the"]
        labels = ["B-Chemical",  "O",        "O"  ]

    After tokenizer (subword level):
        tokens = ["[CLS]", "Nal", "##ox", "##one", "reverses", "the", "[SEP]"]

    Desired output labels:
        labels = [ -100,  B-Chem, -100,   -100,    O,          O,     -100  ]

    Rules:
        [CLS], [SEP], [PAD]  → -100   (special tokens, never predict)
        First subword of word → word's BIO label
        Continuation subword  → -100   (masked from loss)

    Why -100? CrossEntropyLoss(ignore_index=-100) skips those positions.
    Labeling "##ox" as I-Chemical would corrupt the gradient signal because
    the model sees "##ox" in isolation, not as part of "Naloxone". We hide
    those positions so they contribute nothing to training.

    How we detect first-vs-continuation: tokenized_inputs.word_ids(batch_index)
    returns a list parallel to the token sequence. Each element is the integer
    index of the original word that token came from, or None for special tokens.
    If the current word_id equals the previous word_id, it's a continuation.

    Example word_ids for the sentence above:
        [None, 0, 0, 0, 1, 2, None]
         CLS  Nal ##ox ##one  rev  the  SEP
    """
    tokenized_inputs = tokenizer(
        batch["words"],
        is_split_into_words=True,  # tell the tokenizer input is pre-split into words
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,       # return Python lists; Dataset.map handles tensor conversion
    )

    all_labels: List[List[int]] = []

    for batch_idx, word_tags in enumerate(batch["ner_tags"]):
        # word_ids() is indexed by batch position, not by token position.
        word_ids: List[int | None] = tokenized_inputs.word_ids(batch_index=batch_idx)

        aligned_labels: List[int] = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], or [PAD] — masked from loss
                aligned_labels.append(-100)

            elif word_idx != prev_word_idx:
                # First subword token for this word → assign the word's label.
                # .get(..., O) is a safety fallback for any unseen tag strings.
                tag = word_tags[word_idx]
                aligned_labels.append(label2id.get(tag, label2id["O"]))

            else:
                # Continuation subword (e.g., "##ine", "##ase") — masked from loss
                aligned_labels.append(-100)

            prev_word_idx = word_idx

        all_labels.append(aligned_labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# ---------------------------------------------------------------------------
# Debug utility
# ---------------------------------------------------------------------------

def show_alignment(
    words: List[str],
    ner_tags: List[str],
    tokenizer: PreTrainedTokenizerFast,
    label2id: Dict[str, int],
) -> None:
    """Print a side-by-side table of words, subword tokens, and aligned labels.

    Use this in a notebook or REPL to visually verify the alignment is correct
    before committing to a full training run.

    Example output:
        WORD          SUBWORD TOKEN   LABEL
        Naloxone      Nal             B-Chemical
                      ##ox            -100
                      ##one           -100
        reverses      reverses        O
        the           the             O
    """
    example = tokenize_and_align_labels(
        {"words": [words], "ner_tags": [ner_tags]},
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=128,
    )

    id2label = {v: k for k, v in label2id.items()}
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"][0])
    labels = example["labels"][0]

    print(f"\n{'WORD':<20} {'SUBWORD TOKEN':<20} {'LABEL'}")
    print("-" * 60)

    word_ids = tokenizer(
        [words], is_split_into_words=True, max_length=128, truncation=True
    ).word_ids(batch_index=0)

    prev_wid = None
    for token, label_id, wid in zip(tokens, labels, word_ids):
        label_str = id2label.get(label_id, str(label_id)) if label_id != -100 else "-100"
        word_str = words[wid] if wid is not None and wid != prev_wid else ""
        print(f"{word_str:<20} {token:<20} {label_str}")
        if wid is not None:
            prev_wid = wid


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def get_datasets(
    cfg: DataConfig,
) -> Tuple[DatasetDict, PreTrainedTokenizerFast]:
    """Full pipeline: load → preprocess → tokenize.

    Returns:
        tokenized_datasets : DatasetDict with train / validation / test splits,
                             each containing input_ids, attention_mask, labels.
        tokenizer          : needed by DataCollatorForTokenClassification in train.py.
    """
    logger.info("Loading %s (%s)", cfg.dataset_name, cfg.dataset_config)
    raw: DatasetDict = load_dataset(cfg.dataset_name, cfg.dataset_config)

    _, label2id, _ = get_label_info(cfg)
    tokenizer = get_tokenizer(cfg)

    # ---- Stage 1: document → passage expansion -------------------------
    passage_splits: Dict[str, Dataset] = {}
    for split_name in raw:
        logger.info("Processing '%s' split (%d documents)...", split_name, len(raw[split_name]))
        passage_splits[split_name] = build_passage_dataset(raw[split_name], label2id)

    # ---- Stage 2: tokenize and align labels ----------------------------
    # functools.partial bakes tokenizer/label2id/max_length into the function
    # signature so Dataset.map() can call it with just (batch,).
    align_fn = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=cfg.max_length,
    )

    tokenized_splits: Dict[str, Dataset] = {}
    for split_name, ds in passage_splits.items():
        logger.info("Tokenizing '%s' split (%d passages)...", split_name, len(ds))
        tokenized_splits[split_name] = ds.map(
            align_fn,
            batched=True,
            batch_size=256,                        # chunks to bound peak memory during mapping
            remove_columns=["words", "ner_tags"],  # drop string columns; keep tensor columns
            desc=f"Tokenizing {split_name}",
        )

    return DatasetDict(tokenized_splits), tokenizer
