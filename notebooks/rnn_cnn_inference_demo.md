# RNN (BiLSTM) vs TextCNN – Inference Demo (structure-only)

This demo runs inference with **randomly initialized weights** (no trained checkpoints included).
It exists to show **model architecture + preprocessing flow**.

## Quick run
```python
from src.inference import demo_inference
demo_inference()
)


## Embeddings: frozen vs trainable (freeze flag)

In the full project setup, GloVe embeddings can be used either **frozen** or **trainable**.
The switch is controlled by `freeze_embeddings`.

Below is a minimal example that passes a dummy embedding matrix (so the flag is applied in the constructors):

```python
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

# Dummy embedding matrix (stand-in for GloVe)
emb_dim = 100
dummy_embeddings = torch.randn(len(vocab), emb_dim)

# Frozen embeddings
rnn_frozen = BiLSTMClassifier(vocab_size=len(vocab), emb_dim=emb_dim, embeddings=dummy_embeddings, freeze_embeddings=True)
cnn_frozen = TextCNNClassifier(vocab_size=len(vocab), emb_dim=emb_dim, embeddings=dummy_embeddings, freeze_embeddings=True)

# Trainable embeddings
rnn_trainable = BiLSTMClassifier(vocab_size=len(vocab), emb_dim=emb_dim, embeddings=dummy_embeddings, freeze_embeddings=False)
cnn_trainable = TextCNNClassifier(vocab_size=len(vocab), emb_dim=emb_dim, embeddings=dummy_embeddings, freeze_embeddings=False)

print("BiLSTM embedding trainable?", rnn_frozen.embedding.weight.requires_grad, "(frozen expected: False)")
print("BiLSTM embedding trainable?", rnn_trainable.embedding.weight.requires_grad, "(trainable expected: True)")
print("TextCNN embedding trainable?", cnn_frozen.embedding.weight.requires_grad, "(frozen expected: False)")
print("TextCNN embedding trainable?", cnn_trainable.embedding.weight.requires_grad, "(trainable expected: True)")

# Forward pass (untrained demo)
print("BiLSTM logits:", rnn_frozen(X))
print("TextCNN logits:", cnn_frozen(X))


Notes

No dataset files are included in this repository.

No trained checkpoints are included; outputs here are not meaningful predictions.

This notebook-equivalent markdown is meant as a portfolio artifact demonstrating architecture and workflow.
