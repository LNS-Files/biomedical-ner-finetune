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

BC5CDR test set, LoRA BioBERT adapter — **pending retraining with updated config**
(rank=32, class-weighted loss, 10 epochs; numbers below are from the initial run):

| Entity | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Chemical | 0.7067 | 0.7195 | 0.7130 | 4902 |
| Disease | 0.4797 | 0.1500 | 0.2285 | 4254 |
| Micro avg | 0.6589 | 0.4549 | 0.5382 | 9156 |

Overall test metrics (initial run):

- F1: `0.5382`
- Precision: `0.6589`
- Recall: `0.4549`

## Run Locally

Create a Python 3.11 environment, install dependencies, and run:

```bash
python -m src.train --report-to none
python -m src.evaluate --adapter-path results/lora-biobert-bc5cdr
```

Local CPU training is slow. A GPU environment such as Kaggle or Colab is
recommended for full training.

## Run On Kaggle

Use a GPU notebook with Internet enabled:

```python
!git clone https://github.com/LNS-Files/biomedical-ner-finetune.git
%cd biomedical-ner-finetune
!pip install -r requirements.txt
!python -m src.train --report-to none
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
