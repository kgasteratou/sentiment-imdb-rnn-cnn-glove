# Sentiment Analysis – RNN (BiLSTM) vs TextCNN with GloVe

This repository contains compact, public-safe code artifacts for a sentiment analysis project inspired by IMDB-style movie review classification.

## What is included
- Text preprocessing (tokenization, vocabulary, padding)
- Model definitions:
  - **BiLSTM** (RNN-based)
  - **TextCNN**
- **GloVe embeddings** support with a clear **freeze/unfreeze** switch
- Minimal **inference/demo** flow (no dataset files, no checkpoints)

## Embeddings: frozen vs trainable (freeze parameter)
The models support using GloVe embeddings in two modes:
- **Frozen embeddings** (`freeze_embeddings=True`): embedding weights are kept fixed during training.
- **Trainable embeddings** (`freeze_embeddings=False`): embedding weights are fine-tuned during training.

This is controlled via the `freeze_embeddings` parameter in the model constructors:
- `src/models_rnn.py`
- `src/models_cnn.py`

## How to run (Jupyter/Python)
- Minimal demo: `notebooks/rnn_cnn_inference_demo.md`
- Core code: `src/` (preprocessing + model definitions + inference helper)

> Note: This public repository does not ship trained weights; demos focus on architecture and workflow.

## Results (summary)
In the original project setup (IMDB-style sentiment classification), BiLSTM and TextCNN were evaluated with GloVe embeddings under two configurations: **frozen** vs **trainable (fine-tuned)**. Fine-tuning generally improved performance compared to fully frozen embeddings, at the cost of longer training time and higher risk of overfitting. Detailed experiment tables are documented in the accompanying report (not included here).

## Safety / Licensing notes
- No dataset files are uploaded.
- No trained checkpoints are uploaded.
- No private keys or credentials are used.

## Notes
This repository is intended as a portfolio artifact demonstrating NLP model architecture and workflow.
