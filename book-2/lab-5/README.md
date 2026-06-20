# Lab 5 — Benchmark a Server Under Load

**Book 2 · Part V (serving benchmarks, capacity planning)**

---

## Hypothesis

Server latency stays flat as load increases up to a saturation knee, then spikes as the request queue exceeds the server's throughput capacity. The p99 tail latency degrades substantially before the p50 median — the tail is the first warning signal.

## Background

A serving system has a maximum sustainable throughput (its "capacity"). Below capacity, new requests are served almost immediately. As load approaches capacity, the queue begins to fill. Past the knee, queue depth grows without bound and latency diverges. The p99 latency rises faster than p50 because even a short queue causes the slowest requests to wait disproportionately long. Goodput (requests meeting their SLA deadline) peaks at the knee and then collapses even as raw throughput continues rising briefly.

## Setup

```bash
pip install vllm aiohttp matplotlib

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

## Procedure

1. Use an async load driver (`asyncio` + `aiohttp` or vLLM's built-in benchmarking tool) that sends requests at a fixed rate (requests per second).
2. Sweep request rates: e.g., {5, 10, 20, 30, 40, 50, 60, 70, 90} req/s. At each rate, send for 60 seconds and record:
   - p50 TTFT
   - p99 TTFT
   - Throughput (requests completed per second)
3. Plot p50 and p99 latency vs request rate on one chart; plot throughput vs request rate on a second chart.
4. Annotate the knee (the request rate where p99 first exceeds 2× its stable-region value).

**vLLM's built-in benchmark tool (alternative):**
```bash
python -m vllm.benchmarks.benchmark_serving \
    --backend vllm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 200 \
    --request-rate 20
```

## Expected Observations

- **Below the knee:** p50 and p99 are close together and both flat. Throughput scales linearly with request rate.
- **At and above the knee:** p99 spikes sharply first. p50 follows later. Throughput plateaus or drops as the server rejects or timeouts requests.
- **The knee marks safe operating capacity.** Best practice: size the deployment so peak load reaches 60–70% of the knee, leaving headroom for traffic spikes.
- Under high load, TTFT inflates because requests wait in the queue before any GPU work begins — not because prefill or decode slowed down.

## Why This Matters

This lab closes the loop between the theoretical serving architecture (Chapter 13 in Book 1, Chapter 12 in Book 2) and real operational numbers. After running it, you can answer questions that directly affect production decisions: "How many replicas do I need to serve N req/s while keeping p99 TTFT below 2s?" The load-vs-latency curve with an annotated knee is the core artifact of a capacity-planning exercise.

## Variant Experiments

- Repeat the benchmark with prefix caching enabled (`--enable-prefix-caching`) on a shared-system-prompt workload — observe the throughput improvement and latency reduction at the same request rate.
- Compare `--max-num-seqs` settings in vLLM (controls the maximum concurrent sequences the scheduler will batch): observe how a tighter limit lowers peak memory but reduces throughput capacity.
- Introduce request size variability (mix of short and long prompts) and observe how it shifts the knee compared to a uniform workload — variable-length batches are harder to schedule efficiently.
