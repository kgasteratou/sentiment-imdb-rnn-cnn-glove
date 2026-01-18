import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 100, num_classes: int = 2, kernel_sizes=(3,4,5), num_filters: int = 100, padding_idx: int = 0, embeddings=None, freeze_embeddings: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # (B, T, D)
        emb = emb.transpose(1, 2)        # (B, D, T)
        feats = []
        for conv in self.convs:
            c = F.relu(conv(emb))        # (B, F, T-k+1)
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)  # (B, F)
            feats.append(p)
        cat = torch.cat(feats, dim=1)    # (B, F*K)
        logits = self.fc(self.dropout(cat))
        return logits
