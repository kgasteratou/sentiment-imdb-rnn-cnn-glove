import torch
import torch.nn.functional as F
from .preprocess import build_vocab, encode
from .models_rnn import BiLSTMClassifier
from .models_cnn import TextCNNClassifier

def demo_inference():
    # Tiny toy texts (public-safe)
    texts = [
        "I loved this movie. Great acting and story.",
        "Terrible film. Waste of time.",
        "Not bad, but not great either."
    ]

    vocab = build_vocab(texts, min_freq=1)
    max_len = 32

    X = []
    for t in texts:
        ids, _ = encode(t, vocab, max_len=max_len)
        X.append(ids)
    X = torch.tensor(X, dtype=torch.long)

    # NOTE: No trained weights included; this is structure-only demo.
    rnn = BiLSTMClassifier(vocab_size=len(vocab))
    cnn = TextCNNClassifier(vocab_size=len(vocab))

    for name, model in [("BiLSTM", rnn), ("TextCNN", cnn)]:
        logits = model(X)
        probs = F.softmax(logits, dim=1)[:, 1]  # "positive" class prob (arbitrary)
        print(f"{name} (untrained demo) positive probs:", probs.detach().cpu().numpy())

if __name__ == "__main__":
    demo_inference()
