# Biomedical NER Fine-Tuning

Fine-tuning a biomedical named entity recognition model on BC5CDR for chemical
and disease entity extraction.

## Project Goal

This project trains a BioBERT token classification model to identify biomedical
entities in PubMed-style text:

- `Chemical`
- `Disease`

The pipeline loads BC5CDR from Hugging Face BigBio, converts character-level
entity spans into BIO tags, aligns word labels to subword tokens, fine-tunes
BioBERT with LoRA, and evaluates entity-level precision, recall, and F1.

## Stack

- Dataset: `bigbio/bc5cdr`, config `bc5cdr_bigbio_kb`
- Base model: `dmis-lab/biobert-base-cased-v1.2`
- Training method: LoRA via PEFT
- Evaluation: `seqeval`
- Main framework: Hugging Face Transformers

## Results

BC5CDR test set, LoRA BioBERT adapter with rank 32, class-weighted loss, and
10 training epochs:

| Entity    | Precision | Recall |   F1   | Support |
| ---       |   ---:    |  ---:  |  ---:  |  ---:   |
| Chemical  |  0.8093   | 0.9323 | 0.8664 |  4902   |
| Disease   |  0.5576   | 0.8230 | 0.6648 |  4254   |
| Micro avg |  0.6768   | 0.8815 | 0.7657 |  9156   |

Overall test metrics:

- F1: `0.7657`
- Precision: `0.6768`
- Recall: `0.8815`

## Run Locally

Create a Python 3.11 environment, install dependencies, and run:

```bash
python -m src.train
python -m src.evaluate --adapter-path results/lora-biobert-bc5cdr
```

Local CPU training is slow. A GPU environment such as Kaggle or Colab is
recommended for full training.

## Run On Kaggle

Use a GPU notebook with Internet enabled:

```python
!git clone https://github.com/LNS-Files/biomedical-ner-finetune.git
%cd biomedical-ner-finetune
!pip uninstall -y bitsandbytes
!pip install -r requirements.txt
!python -m src.train
!python -m src.evaluate --adapter-path results/lora-biobert-bc5cdr
```

## Project Structure

```text
src/config.py    Hyperparameters and dataset/model configuration
src/data.py      BC5CDR loading, preprocessing, and token-label alignment
src/model.py     BioBERT + LoRA/QLoRA model setup
src/train.py     Training entry point
src/evaluate.py  Test-set evaluation entry point
```
