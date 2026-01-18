import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 100, hidden_dim: int = 128, num_classes: int = 2, padding_idx: int = 0, embeddings=None, freeze_embeddings: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                 # (B, T, D)
        out, _ = self.lstm(emb)                 # (B, T, 2H)
        pooled = out.mean(dim=1)                # simple mean pooling
        logits = self.fc(self.dropout(pooled))  # (B, C)
        return logits
