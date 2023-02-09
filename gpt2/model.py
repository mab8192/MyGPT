import math
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias parameter"""
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x0: torch.Tensor):
        return F.layer_norm(x0, self.weight.shape, self.weight, self.bias, eps=1e-6)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        embed_dim = config["embed_dim"]
        bias = config["sa_bias"]
        dropout = config["sa_dropout"]
        block_size = config["block_size"]

        self.n_heads = config["n_heads"]
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Layers for queries, keys, and values rolled into a single batch
        self.c_attn = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Buffer for masked attention
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x0: torch.Tensor):
        B, T, C = x0.shape

        q, k, v = self.c_attn(x0).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # Will be shape (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # Will be shape (B, n_heads, T, head_size)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # Will be shape (B, n_heads, T, head_size)

        attn = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.shape[-1]))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # recombine outputs from all heads

        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    """Simple multilayer perceptron"""
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        embed_dim = config["embed_dim"]
        bias = config["mlp_bias"]
        dropout = config["mlp_dropout"]

        self.fc = nn.Linear(embed_dim, 4*embed_dim, bias=bias)
        self.proj = nn.Linear(4*embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0: torch.Tensor):
        x = self.fc(x0)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        embed_dim = config["embed_dim"]
        bias = config["ln_bias"]

        self.ln1 = LayerNorm(embed_dim, bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(config)

    def forward(self, x0: torch.Tensor):
        x = self.attn(self.ln1(x0)) + x0
        x = self.mlp(self.ln2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        # Build the model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config["vocab_size"], config["embed_dim"]),
            wpe = nn.Embedding(config["block_size"], config["embed_dim"]),
            drop = nn.Dropout(config["dropout"]),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])]),
            ln_f = LayerNorm(config["embed_dim"], config["ln_bias"])
        ))

        self.lm_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply scaled initialization, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["n_layers"]))

    def forward(self, x0: torch.Tensor, targets: torch.Tensor = None):
        device = x0.device
        b, t = x0.shape

        assert t <= self.config["block_size"], \
            f"Cannot forward sequence of length {t}, cannot be longer than {self.config['block_size']}."

        # Create positional encodings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Forward through GPT
        tok_emb = self.transformer.wte(x0)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Compute loss if targets are provided
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            x_cond = x if x.shape[1] <= self.config["block_size"] else x[:, -self.config["block_size"]:]  # Truncate if necessary
            logits, _ = self(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)  # cat the new token with the previous tokens in the time dimension

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        raise NotImplementedError

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
