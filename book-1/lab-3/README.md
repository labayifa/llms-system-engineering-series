# Lab 3 — Compare Temperatures

**Book 1 · Part IV (temperature, top_p, entropy)**

---

## Hypothesis

As temperature rises, output diversity rises and determinism falls. `top_p` only meaningfully constrains outputs once the distribution is already flat (high temperature) — at low temperature the nucleus already contains nearly all the probability mass.

## Background

Temperature T scales the logits before softmax: `logits / T`. At T→0 the distribution collapses to a one-hot (greedy decoding). At T=1 the raw model distribution is used. At T>1 the distribution flattens toward uniform. Nucleus sampling (top_p) selects the smallest set of tokens whose cumulative probability exceeds p and discards the rest — it is an adaptive top-k. When T is low, the top few tokens already hold almost all the mass, so top_p=0.9 changes nothing. When T is high, the mass is spread across thousands of tokens, and top_p=0.9 meaningfully trims the incoherent tail.

## Setup

```bash
pip install vllm requests

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

## Procedure

1. Choose one open-ended prompt (e.g., "List three creative uses for an old bicycle:").
2. For each temperature T ∈ {0, 0.2, 0.5, 0.8, 1.2}, generate 20 completions.
3. Record the **unique fraction**: `len(set(outputs)) / 20`.
4. Repeat the T=1.2 run with `top_p=0.9` and compare unique fraction and qualitative coherence.
5. Optionally, for each T compute the average token-level entropy from the logprobs vLLM can return (`logprobs=5`).

## Expected Observations

- **Unique fraction ≈ 0 at T=0** — all 20 completions are identical (greedy decoding is deterministic).
- **Unique fraction rises toward 1 as T increases** — at T=1.2 nearly every completion differs.
- **Adding top_p=0.9 at T=1.2 reduces incoherence** but does not dramatically change the unique fraction — because it trims only the very low-probability tail.
- Average token entropy should rise monotonically with temperature.

## Why This Matters

Choosing the right temperature is task-specific:

| Task type | Temperature | Reason |
|-----------|------------|--------|
| JSON extraction, SQL, classification | 0 – 0.1 | Single correct answer; diversity is noise |
| Factual Q&A, summarization | 0.2 – 0.5 | Some flexibility, but grounding matters |
| Creative writing, brainstorming | 0.7 – 1.0 | Diversity is the point |
| Adversarial red-teaming, data augmentation | ≥ 1.0 | Maximum exploration |

Running this lab first-hand makes the tradeoffs visceral rather than abstract.

## Variant Experiments

- Vary `top_k` instead of `top_p` at high temperature and compare outputs — `top_k` is a fixed window (not adaptive), so it behaves differently when the distribution changes shape.
- Test a structured prompt (e.g., "Return only a JSON object with keys 'name' and 'age'") at T=0 vs T=0.8 — observe the format-compliance rate drop as temperature rises.
- Plot the distribution of completion lengths across temperatures — high temperature often produces longer, more rambling outputs.
