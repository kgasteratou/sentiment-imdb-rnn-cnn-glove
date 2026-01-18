# Sentiment Analysis – RNN (BiLSTM) vs TextCNN with GloVe

This repository contains compact, public-safe code artifacts for a sentiment analysis project
inspired by IMDB-style movie review classification.

## What is included
- Text preprocessing (tokenization, padding)
- Model definitions:
  - **BiLSTM** (RNN-based)
  - **TextCNN**
- **GloVe embeddings** support (frozen vs trainable)
- Minimal inference demo (no dataset files)

## Embeddings: frozen vs trainable (freeze parameter)
The models support using GloVe embeddings in two modes:
- **Frozen embeddings** (`freeze_embeddings=True`): embedding weights are kept fixed during training.
- **Trainable embeddings** (`freeze_embeddings=False`): embedding weights are fine-tuned during training.

This is controlled via the `freeze_embeddings` parameter in the model constructors (see `src/models_rnn.py` and `src/models_cnn.py`).

## How to run (Jupyter/Python)
- See `notebooks/rnn_cnn_inference_demo.md` for a minimal inference demo.
- Core code lives in `src/`.

## Results (summary)
In the original project setup (IMDB-style sentiment classification), both BiLSTM and TextCNN were evaluated with GloVe embeddings under two configurations: **frozen** vs **trainable (fine-tuned)**. Fine-tuning embeddings generally improved performance compared to fully frozen embeddings, at the cost of longer training time and higher risk of overfitting. Detailed results and experiment tables are documented in the accompanying report (not included in this public repository).

## Safety / Licensing notes
- No dataset files are uploaded.
- No trained checkpoints are uploaded.
- This repository focuses on architecture, preprocessing, and reproducible code structure.

## Notes
This repository is intended as a portfolio artifact demonstrating NLP model architecture and workflow.

