# RNN (BiLSTM) vs TextCNN – Inference Demo (structure-only)

This demo runs inference with **randomly initialized weights** (no trained checkpoints included).
It exists to show **model architecture + preprocessing flow**.

## Quick run
```python
from src.inference import demo_inference
demo_inference()
)


Embeddings: frozen vs trainable (freeze flag)

In the full project setup, GloVe embeddings can be used either frozen or trainable.
The switch is controlled by freeze_embeddings.

Below is a minimal example showing how the flag is passed to each model:

import torch
from src.preprocess import build_vocab, encode
from src.models_rnn import BiLSTMClassifier
from src.models_cnn import TextCNNClassifier

texts = [
    "I loved this movie. Great acting and story.",
    "Terrible film. Waste of time."
]

vocab = build_vocab(texts, min_freq=1)
X = torch.tensor([encode(t, vocab, max_len=32)[0] for t in texts], dtype=torch.long)

# Frozen embeddings (during training embeddings would remain fixed)
rnn_frozen = BiLSTMClassifier(vocab_size=len(vocab), freeze_embeddings=True)
cnn_frozen = TextCNNClassifier(vocab_size=len(vocab), freeze_embeddings=True)

# Trainable embeddings (during training embeddings would be fine-tuned)
rnn_trainable = BiLSTMClassifier(vocab_size=len(vocab), freeze_embeddings=False)
cnn_trainable = TextCNNClassifier(vocab_size=len(vocab), freeze_embeddings=False)

# Forward pass (untrained demo)
print("BiLSTM logits (frozen):", rnn_frozen(X))
print("TextCNN logits (frozen):", cnn_frozen(X))
print("BiLSTM logits (trainable):", rnn_trainable(X))
print("TextCNN logits (trainable):", cnn_trainable(X))

Notes

No dataset files are included in this repository.

No trained checkpoints are included; outputs here are not meaningful predictions.

This notebook-equivalent markdown is meant as a portfolio artifact demonstrating architecture and workflow.
