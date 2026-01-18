import re
from typing import List, Dict, Tuple

def basic_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq = {}
    for t in texts:
        for tok in basic_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def encode(text: str, vocab: Dict[str, int], max_len: int = 64) -> Tuple[List[int], int]:
    toks = basic_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks][:max_len]
    length = len(ids)
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids, length
