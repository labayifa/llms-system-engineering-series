# Lab 1 — Measure TTFT

**Book 1 · Part I (prefill) and Part VII (request lifecycle)**

---

## Hypothesis

TTFT (Time to First Token) scales with input length because it is dominated by the prefill phase, which processes all input tokens in parallel and costs O(N²) in attention. Output length should barely affect TTFT because TTFT is measured at the first decode step, before any output tokens are generated.

## Background

Prefill processes the entire prompt in one parallel forward pass — compute-bound, scales with input length. Decode generates one token per step — memory-bandwidth-bound, sequential. TTFT is the wall-clock time from request submission to the first token in the response. It is dominated by prefill once the request reaches the model, but under heavy load the queue wait time can dominate entirely.

## Setup

```bash
# Start a vLLM server (in a separate terminal)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# Install dependencies
pip install vllm requests
```

## Procedure

1. Hold output length fixed at 50 tokens. Run prompts of 100, 500, 2000, and 8000 input tokens. Record TTFT for each.
2. Hold input fixed at 500 tokens. Vary `max_tokens` from 50 to 500. Record TTFT for each.
3. Plot TTFT vs input token count and TTFT vs output token count on the same figure.

Run the starter script:
```bash
python lab.py
```

## Expected Observations

- **TTFT rises clearly with input length** — roughly linearly (or faster for very long contexts, O(N²) attention).
- **TTFT is nearly flat as output length changes** — confirming TTFT is a prefill metric.
- The two plots have visually different slopes — steep vs flat — which directly demonstrates why TTFT and tokens/sec are measured separately.

## Why This Matters

Understanding that TTFT ≠ E2E latency and TTFT ≠ tokens/sec is the first step to diagnosing production latency. A high TTFT with normal tokens/sec means a prefill problem (long prompts). A normal TTFT with low tokens/sec means a decode problem (memory bandwidth). Under queue saturation, both worsen but for different reasons.

## Variant Experiments

- Add a concurrent-request sweep: send 1, 4, 8, 16 parallel requests and see how queue delay inflates TTFT.
- Enable prefix caching (`--enable-prefix-caching`) and repeat with a shared system prompt — TTFT should collapse for cache-hitting requests.
