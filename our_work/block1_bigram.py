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


@app.cell
def _(mo):
    mo.md("""
    # Self-Attention: Building Intuition

    Three versions of weighted averaging over past context, leading up to self-attention.
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)

    # Version 1: for-loop bag of words
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1, :]  # (t, C)
            xbow[b, t] = xprev.mean(dim=0)

    print("x[0]:")
    print(x[0])
    print("\nxbow[0] (bag of words via for loop):")
    print(xbow[0])
    return T, x


@app.cell
def _(T, torch, x):
    # Version 2: matrix multiply trick
    _wei = torch.tril(torch.ones(T, T))
    _wei = _wei / _wei.sum(dim=1, keepdim=True)
    _xbow2 = _wei @ x  # (T, T) @ (B, T, C) -> (B, T, C)
    print("_wei:")
    print(_wei)
    print("\n_xbow2[0] (matrix multiply trick):")
    print(_xbow2[0])
    return


@app.cell
def _(F, T, torch, x):
    # Version 3: softmax with masked fill
    _tril = torch.tril(torch.ones(T, T))
    _wei = torch.zeros((T, T))
    _wei = _wei.masked_fill(_tril == 0, float('-inf'))
    _wei = F.softmax(_wei, dim=1)
    _xbow3 = _wei @ x  # (B, T, C)
    print("_wei:")
    print(_wei)
    print("\n_xbow3[0] (softmax + masked fill):")
    print(_xbow3[0])
    return


@app.cell
def _(mo):
    mo.md("""
    # Version 4: Single Head of Self-Attention

    Now we implement actual self-attention with learned key, query, and value projections.
    """)
    return


@app.cell
def _(F, nn, torch):
    class Head(nn.Module):
        def __init__(self, n_embd, head_size, block_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)   # (B, T, head_size)
            q = self.query(x) # (B, T, head_size)

            # compute attention scores
            head_size = k.shape[-1]
            wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)  # (B, T, T)

            # weighted aggregation of values
            v = self.value(x)  # (B, T, head_size)
            out = wei @ v  # (B, T, head_size)
            return out

    return (Head,)


@app.cell
def _(Head, torch):
    torch.manual_seed(1337)
    _B, _T, _C = 4, 8, 32
    _head_size = 16
    _x = torch.randn(_B, _T, _C)

    _head = Head(n_embd=_C, head_size=_head_size, block_size=_T)
    _out = _head(_x)
    print(f"Input shape:  {_x.shape}")
    print(f"Output shape: {_out.shape}")
    print(f"Expected:     torch.Size([{_B}, {_T}, {_head_size}])")
    return


@app.cell
def _(mo):
    mo.md("""
    # Self-Attention Language Model

    Upgrading the bigram model: token embeddings + positional embeddings + single-head self-attention.
    """)
    return


@app.cell
def _():
    sa_block_size = 8
    sa_n_embd = 32
    sa_head_size = 32
    return sa_block_size, sa_head_size, sa_n_embd


@app.cell
def _(F, Head, nn, sa_block_size, sa_head_size, sa_n_embd, torch, vocab_size):
    class SelfAttentionLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, sa_n_embd)
            self.position_embedding_table = nn.Embedding(sa_block_size, sa_n_embd)
            self.sa_head = Head(sa_n_embd, sa_head_size, sa_block_size)
            self.lm_head = nn.Linear(sa_head_size, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
            x = tok_emb + pos_emb  # (B, T, n_embd)
            x = self.sa_head(x)  # (B, T, head_size)
            logits = self.lm_head(x)  # (B, T, vocab_size)

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
                idx_cond = idx[:, -sa_block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]  # (B, C)
                probs = F.softmax(logits, dim=-1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            return idx

    return (SelfAttentionLM,)


@app.cell
def _(SelfAttentionLM, torch):
    torch.manual_seed(1337)
    sa_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sa_model = SelfAttentionLM()
    sa_model = sa_model.to(sa_device)
    print(f"Parameters: {sum(p.numel() for p in sa_model.parameters()) / 1e3:.1f}K")
    return sa_device, sa_model


@app.cell
def _(get_batch, sa_model, torch):
    sa_max_iters = 3000
    sa_eval_interval = 300
    sa_learning_rate = 1e-3
    sa_eval_iters = 200

    _optimizer = torch.optim.AdamW(sa_model.parameters(), lr=sa_learning_rate)

    @torch.no_grad()
    def _estimate_loss():
        _out = {}
        sa_model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(sa_eval_iters)
            for k in range(sa_eval_iters):
                X, Y = get_batch(split)
                _, loss = sa_model(X, Y)
                losses[k] = loss.item()
            _out[split] = losses.mean()
        sa_model.train()
        return _out

    for _iter in range(sa_max_iters):
        if _iter % sa_eval_interval == 0:
            _losses = _estimate_loss()
            print(f"step {_iter}: train loss {_losses['train']:.4f}, val loss {_losses['val']:.4f}")

        _xb, _yb = get_batch('train')
        _logits, _loss = sa_model(_xb, _yb)
        _loss.backward()
        _optimizer.step()
        _optimizer.zero_grad(set_to_none=True)

    _losses = _estimate_loss()
    print(f"Final: train loss {_losses['train']:.4f}, val loss {_losses['val']:.4f}")
    return


@app.cell
def _(decode, sa_device, sa_model, torch):
    _context = torch.zeros((1, 1), dtype=torch.long, device=sa_device)
    _generated = sa_model.generate(_context, max_new_tokens=500)
    print(decode(_generated[0].tolist()))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
