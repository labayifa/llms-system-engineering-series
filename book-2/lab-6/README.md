# Lab 6 — Measure Speculative Decoding

**Book 2 · Part III (speculative decoding internals)**

---

## Hypothesis

Speculative decoding speedup is directly proportional to the draft model's acceptance rate, which in turn depends on how closely the draft model's distribution matches the target model's distribution. A stronger draft model yields higher acceptance and higher tokens/sec, up to the point where the draft model itself becomes the bottleneck.

## Background

Speculative decoding uses a small, fast **draft model** to propose k tokens, then verifies all k simultaneously in a single forward pass of the large **target model**. Tokens accepted by the target's rejection-sampling procedure are kept; the first rejected token is resampled. Expected tokens accepted per step is `k × α`, where α is the acceptance rate. When α is high, near-k tokens are generated per target forward pass, giving a speedup of roughly `k × α / 1` over standard decode. When α is low (draft and target distributions differ), most proposals are rejected, and the overhead of running both models nets negative.

## Setup

```bash
pip install vllm matplotlib

# vLLM supports speculative decoding natively:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \         # target model
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \  # draft model
    --num-speculative-tokens 5 \
    --port 8000
```

## Procedure

1. **Baseline:** benchmark the target model alone (no speculative decoding) — record tokens/sec and p50 TTFT at a fixed request rate.
2. **Speculative decoding, weak draft:** use a draft model with fewer parameters or a different family than the target. Record tokens/sec, p50 TTFT, and the `spec_decode_acceptance_rate` metric exposed by vLLM.
3. **Speculative decoding, strong draft:** switch to a draft model from the same family as the target (e.g., 1B draft for a 7B target of the same series). Record the same metrics.
4. Sweep `--num-speculative-tokens` from 1 to 8 with the strong draft. Plot acceptance rate and tokens/sec vs k.
5. Compute the theoretical speedup `k × α` and compare to the observed speedup.

## Expected Observations

- **Weak draft:** acceptance rate < 0.5; tokens/sec may be *lower* than the baseline because draft inference overhead exceeds the benefit.
- **Strong draft:** acceptance rate > 0.7; tokens/sec clearly exceeds the baseline — 1.5×–2× is typical.
- **Increasing k** raises tokens/sec up to the point where the draft model becomes the throughput bottleneck (its forward passes take longer than the target gains).
- The theoretical speedup formula `k × α` tracks the observed speedup closely — demonstrating the model is well-calibrated.
- TTFT is unaffected or slightly worse with speculative decoding (the draft introduces prefill overhead).

## Why This Matters

Speculative decoding is the most technique-dependent optimization in LLM serving: it works well on some workloads and not at all on others. This lab develops the intuition for when to use it. High acceptance requires that the draft and target have similar output distributions — which holds when they share training data and architecture, and breaks when they don't. Code generation (predictable, constrained vocabulary) tends to have high acceptance; open-ended chat tends to have lower acceptance. After running this lab, you can evaluate whether speculative decoding is worth deploying for your specific workload.

## Variant Experiments

- Test speculative decoding on structured output tasks (JSON generation, code completion) vs open-ended chat and compare acceptance rates — structured tasks often show much higher acceptance.
- Vary the target model temperature and observe its effect on acceptance rate: at T=0 (greedy), the target is maximally deterministic and acceptance is highest.
- Profile both the draft and target model with Nsight Compute during speculative decoding and verify that the target's utilization increases (more work per unit time) while the draft's utilization sets the throughput ceiling.
