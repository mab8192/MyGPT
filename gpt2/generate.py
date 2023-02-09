from typing import Callable
from argparse import ArgumentParser

import torch
from model import GPT


def generate(model: GPT, prompt: str, encode: Callable, decode: Callable):
    inp = torch.Tensor(encode(prompt)).view(1, -1).long()
    return decode(model.generate(inp, max_new_tokens=500)[0].tolist())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", default="gpt2_3000.pth")

    args = parser.parse_args()

    block_size = 256
    embed_dim = 384
    n_heads = 6
    n_layers = 6
    dropout = 0.2

    with open("../datasets/starwars.txt") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    model_config = dict(
    vocab_size = vocab_size,
    block_size = block_size,
    embed_dim = embed_dim,
    dropout = dropout,
    n_layers = n_layers,
    n_heads = n_heads,
    ln_bias = True,
    sa_bias = True,
    sa_dropout = dropout,
    mlp_bias = True,
    mlp_dropout = dropout
)

    model = GPT(model_config)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    while True:
        prompt = input("Enter a prompt: ")
        if prompt == "quit": break

        response = generate(model, prompt, encode, decode)
        print(response)
        print()
