import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    return F, mo, nn, torch


@app.cell
def _(mo):
    mo.md("""
    # Block 1: Data & Bigram Baseline

    Building a character-level language model on Tiny Shakespeare.
    """)
    return


@app.cell
def _():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Length of dataset: {len(text)} characters")
    print(f"\nFirst 200 characters:\n{text[:200]}")
    return (text,)


@app.cell
def _(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary: {''.join(chars)}")
    print(f"Vocab size: {vocab_size}")
    return chars, vocab_size


@app.cell
def _(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    test_str = "hello"
    encoded = encode(test_str)
    decoded = decode(encoded)
    print(f"encode('{test_str}') = {encoded}")
    print(f"decode({encoded}) = '{decoded}'")
    return decode, encode


@app.cell
def _(encode, text, torch):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data


@app.cell
def _():
    batch_size = 32
    block_size = 8
    return batch_size, block_size


@app.cell
def _(batch_size, block_size, device, torch, train_data, val_data):
    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    _xb, _yb = get_batch("train")
    print(f"Input shape: {_xb.shape}, Target shape: {_yb.shape}")
    print(f"\nFirst sequence input:  {_xb[0]}")
    print(f"First sequence target: {_yb[0]}")
    return (get_batch,)


@app.cell
def _(F, nn, torch):
    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx, targets=None):
            logits = self.token_embedding_table(idx)  # (B, T, C)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                logits, loss = self(idx)
                logits = logits[:, -1, :]  # (B, C)
                probs = F.softmax(logits, dim=1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            return idx

    return (BigramLanguageModel,)


@app.cell
def _(BigramLanguageModel, torch, vocab_size):
    torch.manual_seed(1337)
    model = BigramLanguageModel(vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    return device, m, model


@app.cell
def _(get_batch, model, torch):
    max_iters = 3000
    eval_interval = 300
    learning_rate = 1e-2

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    eval_iters = 200

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    losses = estimate_loss()
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    return


@app.cell
def _(decode, device, m, torch):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = m.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))
    return


if __name__ == "__main__":
    app.run()
