"""
A bigram language model learns the statistics of bigrams, which are 2
letter combinations, and uses the learned representation to predict characters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding(idx)  # (B, T, C) batch x time x channels

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range (max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Take the last element in the time dimension
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


