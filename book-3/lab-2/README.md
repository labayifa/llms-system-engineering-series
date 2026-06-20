# Lab 2 — Compare Positional Encodings

**Book 3 · Part A, Chapter 1 (RoPE, relative vs absolute, extrapolation)**

---

## Hypothesis

Models trained with learned absolute positional embeddings degrade sharply on sequences longer than their training length, because positions beyond the training range have undefined embeddings. Models using RoPE (Rotary Position Embedding) degrade more gracefully, because RoPE encodes *relative* position — a property that generalizes to unseen lengths.

## Background

**Absolute positional embeddings:** each position index maps to a unique learned vector. The model sees positions 0 through T_max during training. At inference on position T_max+100, the embedding table has no entry — behavior is undefined (and typically catastrophic).

**RoPE:** applies a rotation matrix to Q and K vectors such that the inner product Q·K depends only on the *relative* offset between positions. The rotation angle is a function of the offset, not the absolute position. A model trained on sequences up to 2048 tokens still has a well-defined geometric interpretation for position 3000 — the relative offset between any two tokens within a window is handled the same way.

## Setup

```bash
pip install torch transformers matplotlib

# This lab trains small toy models — no large GPU needed
# A single RTX 3090 or even CPU is sufficient for the toy scale
```

## Procedure

1. Define a tiny transformer (2 layers, 128 hidden dim, 4 heads) in two variants:
   - **Absolute:** standard `nn.Embedding(max_length, d_model)` added to token embeddings. Set `max_length = 128`.
   - **RoPE:** apply RoPE rotations to Q and K in each attention head. No positional embedding table.

2. Train both models on the same task requiring position sensitivity (e.g., copy task: reproduce the input sequence verbatim, or a simple arithmetic task where position encodes operand order). Use sequence length 128.

3. Evaluate both models at lengths 128 (in-distribution), 192, 256, 384, 512 (all out-of-distribution). Record task accuracy at each length.

4. Plot accuracy vs sequence length for both models on the same axes.

## Expected Observations

- **At length 128 (training length):** both models perform comparably.
- **Past length 128:** the absolute model's accuracy drops sharply — often to near-chance — because trained position embeddings don't exist for the new indices.
- **RoPE degrades more gradually:** some degradation is expected at very long extrapolation ratios (e.g., 4× training length), but the model retains meaningful accuracy well beyond its training context.
- The degradation profile is a concrete illustration of why modern production models (Llama, Qwen, Mistral) have universally adopted RoPE.

## Why This Matters

Context-length extrapolation is one of the most practically important properties of a production model. A model that fails at 2× its training context is useless for long-document tasks even if its training context was nominally "long." Understanding *why* RoPE extrapolates — relative geometry vs absolute table lookup — gives you the foundation to evaluate context-extension techniques (YaRN, LongRoPE, ALiBi) and to choose or configure models for long-context workloads.

## Variant Experiments

- Compare RoPE with **ALiBi** (Attention with Linear Biases): ALiBi adds a learned linear bias to attention scores proportional to distance. Train a third tiny model variant and add it to the plot.
- Apply **YaRN** or **RoPE scaling** (`rope_scaling` in HuggingFace config) to the RoPE model and observe whether it extends the usable context further — this is the extrapolation technique used to take a 4k-context model to 128k.
- Visualize the attention patterns (attention weight matrices) at positions within and beyond the training range for both models — the absolute model's patterns become incoherent past T_max in a way that is visually clear.
