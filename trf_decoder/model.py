"""
This transformer implementation only contains the decoder. It's a simple model that
just predicts the character in a sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 256
embed_dim = 128
n_heads = 4
n_layers = 4
dropout = 0.2


# ----- Define Model -----
class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei@v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=2)
        out = self.dropout(self.proj(out))
        return out


class FF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head_size = embed_dim // n_heads
        self.sa = MultiHeadAttention(head_size)
        self.ff = FF()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x


class TransformerDecoderLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)  # Final layer norm
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # (B, T, embed_dim) batch x time x channels
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            idx_con = idx[:, -block_size:]  # Truncate to ensure we never pass in more than `block_size` tokens
            logits, loss = self(idx_con)
            logits = logits[:, -1, :] # Take the last element in the time dimension
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
