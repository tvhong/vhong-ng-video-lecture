## Lecture: Let's Build GPT from Scratch

**Video**: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h56m)
**Due**: TBD
**Est. Hours**: 10h

### Key Concepts

- Tokenization (character-level vs subword/BPE)
- Data loader: batches of chunks, block_size as max context
- Bigram baseline → self-attention → full Transformer
- Self-attention: queries, keys, values; scaled dot-product; causal masking via triangular matrix
- The matrix multiply trick: lower triangular weighted aggregation = efficient past-context averaging
- Multi-headed self-attention: multiple independent communication channels, concatenated
- Feed-forward layers: per-token computation after communication
- Residual connections (skip connections): gradient superhighway, blocks fork off and project back via addition
- LayerNorm: pre-norm formulation (normalize before attention/FFN, not after)
- Dropout: regularization by randomly zeroing activations
- Positional embeddings: attention has no notion of space, must encode position explicitly
- Encoder vs decoder vs encoder-decoder Transformers; self-attention vs cross-attention
- nanoGPT walkthrough; pretraining vs fine-tuning (SFT + RLHF)

### What We'll Build

A decoder-only Transformer trained on Tiny Shakespeare (~1MB text). Character-level language model that generates Shakespeare-like text. Start from bigram baseline, progressively add self-attention, multi-head attention, feed-forward, residual connections, layer norm, and dropout.

### Time Log

| Date       | Hours                | What                                                                                  |
| ---------- | -------------------- | ------------------------------------------------------------------------------------- |
| 2026-04-06 | 1.5h (3:30–5:00 PM)  | Block 1: Data & Bigram Baseline (items 1-4 complete)                                  |
| 2026-04-07 | 1.5h (8:00–9:30 PM)  | Block 2: Self-Attention versions 1-3 (items 5-7 complete)                             |
| 2026-04-08 | 1.5h (9:30–11:00 PM) | Block 2: Watched single-head self-attention & positional embeddings, Q/K/V discussion |
| 2026-04-09 | 0.75h                | Block 2: Watched video up to 1:16:00                                                  |

#### Block 1: Data & Bigram Baseline (0:00–38:01, ~1.5h implement)

**Watch** intro through bigram model training. Then implement:

1. [x] Read Tiny Shakespeare, build character-level tokenizer (encode/decode), train/val split
2. [x] Implement data loader: sample random chunks of block_size, batch them into (B, T) tensors
3. [x] Implement bigram language model (nn.Module): token embedding → logits, cross-entropy loss, generate function
4. [x] Train bigram model, verify loss decreases from ~4.87 toward ~2.5

#### Block 2: Self-Attention (38:01–1:10:01, ~3h implement)

**Watch** the matrix multiply trick (versions 1-3), then single-head self-attention. This is the core of the lecture. Then implement:

5. [x] Implement version 1: weighted averaging of past context with for loops (bag of words)
6. [x] Implement version 2: lower triangular matrix multiply trick (tril + ones, batched matmul)
7. [x] Implement version 3: masked_fill with -inf + softmax (this becomes the attention pattern)
8. [ ] Implement single head of self-attention: key, query, value projections; scaled dot-product; causal mask
9. [ ] Add positional embeddings; plug self-attention head into the model; train and verify improvement

#### Block 3: Multi-Head Attention & Feed-Forward (1:10:01–1:28:00, ~1.5h implement)

**Watch** multi-head attention, feed-forward, and Transformer blocks. Then implement:

10. [ ] Implement multi-head attention: multiple heads in parallel, concatenate outputs
11. [ ] Implement feed-forward network (Linear → ReLU → Linear) applied per-token after attention
12. [ ] Create Transformer Block class: self-attention → feed-forward, stack multiple blocks

#### Block 4: Residual Connections, LayerNorm, Dropout (1:28:00–1:42:00, ~1.5h implement)

**Watch** residual connections, layer norm, dropout, and scaling up. Then implement:

13. [ ] Add residual connections: x = x + self_attn(x), x = x + ffwd(x)
14. [ ] Add projection layers back into residual pathway (linear after attention concat, linear in FFN)
15. [ ] Add LayerNorm (pre-norm formulation: normalize before attention and FFN)
16. [ ] Add Dropout before residual connections and after softmax in attention
17. [ ] Scale up: larger embedding dim, more heads, more layers; train and push val loss below 2.0

#### Block 5: Encoder/Decoder Discussion & nanoGPT (1:42:00–end, watch only)

**Watch** the encoder vs decoder discussion, nanoGPT walkthrough, and ChatGPT/GPT-3 comparison. No implementation needed.

18. [ ] Understand: decoder-only (causal mask) vs encoder (no mask) vs encoder-decoder (cross-attention)
19. [ ] Understand: pretraining (document completer) vs fine-tuning (SFT → reward model → RLHF)
