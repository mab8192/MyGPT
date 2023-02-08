import torch

from model import BigramLanguageModel

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 8
batch_size = 32
max_iters = 3000
eval_iters = 50
eval_every = 300

# ----- Load and prepare data -----
with open("input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.Tensor(encode(text)).long()

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

model = BigramLanguageModel(vocab_size)


# ----- Train Model -----
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(max_iters):
    xb, yb = get_batch('train')

    if (step + 1) % eval_every == 0:
        losses = estimate_loss()
        print(f"Epoch {step+1}: train {losses['train']} val {losses['val']}")

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ----- Test Model -----
inp = torch.zeros(1, 1).long()
print(decode(model.generate(inp, max_new_tokens=500)[0].tolist()))