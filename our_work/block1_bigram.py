import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    return mo, torch


@app.cell
def _(mo):
    mo.md("""
    # Block 1: Data & Bigram Baseline

    Building a character-level language model on Tiny Shakespeare.
    """)
    return


@app.cell
def _():
    with open('input.txt', 'r', encoding='utf-8') as f:
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
    return (chars,)


@app.cell
def _(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return ''.join([itos[i] for i in l])

    test_str = "hello"
    encoded = encode(test_str)
    decoded = decode(encoded)
    print(f"encode('{test_str}') = {encoded}")
    print(f"decode({encoded}) = '{decoded}'")
    return (encode,)


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
def _(batch_size, block_size, torch, train_data, val_data):
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    xb, yb = get_batch('train')
    print(f"Input shape: {xb.shape}, Target shape: {yb.shape}")
    print(f"\nFirst sequence input:  {xb[0]}")
    print(f"First sequence target: {yb[0]}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
