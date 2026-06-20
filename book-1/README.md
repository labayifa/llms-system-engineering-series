# Book 1 — LLM Inference, Sampling, and Context Engineering

> *An illustrated field guide for ML systems engineers*

**Author:** Carmel Prosper Sagbo — CyLab-Africa / Upanzi Network, CMU Africa  
**Part of:** [LLM Systems Engineering: A Three-Volume Series](../README.md)

---

## Summary

Book 1 answers a single question: **how do you use and serve a language model well?** It treats the GPU, kernels, and training process as boxes that simply work, and focuses entirely on what happens from the moment a prompt arrives to the moment the last token streams back.

The book is structured as eight parts that build on one another. Every mechanism gets a diagram before the explanation, and a plain-language summary for every hard idea. The five hands-on labs at the end turn the concepts into measurements.

### What you learn

**Part I — Transformer Inference**  
How the attention mechanism works (Q, K, V, scaled dot-product, multi-head). The two distinct phases of inference — prefill (parallel, compute-bound) and decode (sequential, memory-bound) — and why they have fundamentally different performance profiles. Key latency metrics: TTFT, inter-token latency, tokens/sec, E2E latency.

**Part II — Hardware and Systems**  
Why decode is slow: arithmetic intensity, HBM bandwidth, and the memory-bandwidth bottleneck. The KV cache formula and how much VRAM a long context actually consumes. Grouped Query Attention (GQA) and its 4× KV reduction. Mixture of Experts: active vs total parameters, routing, load balancing, and why MoE saves compute but not memory.

**Part III — Optimization**  
Six techniques that attack memory bandwidth or the sequential decode bottleneck: KV caching, prefix caching, continuous batching, quantization (INT8/INT4), FlashAttention, speculative decoding, PagedAttention, and prefill–decode disaggregation.

**Part IV — Sampling and Decoding**  
How temperature reshapes the logit distribution and why T=0 vs T≥0.8 matters. Nucleus sampling (top_p), top_k, repetition penalties, and how temperature and top_p interact. The task–temperature matrix.

**Part V — Context Engineering**  
Context window budgeting. The lost-in-the-middle effect and why retrieval ordering matters. Context compression. RAG architecture end-to-end: ingestion (chunking, embedding, vector DB) and query (retrieval, reranking, generation). Embeddings: bi-encoders vs cross-encoders, cosine similarity, choosing a model. Agent context engineering.

**Part VI — Evaluation**  
Offline metrics (Exact Match, F1, ROUGE, BLEU, BERTScore). LLM-as-Judge: pairwise ranking, rubric scoring, and judge biases to control for. RAG-specific metrics: context precision, context recall, answer faithfulness, answer relevance. Reading benchmarks critically (MMLU, GPQA, HumanEval, SWE-bench).

**Part VII — Production Architectures**  
What a serving engine does: queue management, admission control, batch formation, KV-cache memory management. The full request lifecycle from user to streaming response. vLLM, TGI, SGLang, TensorRT-LLM. Tensor parallelism vs pipeline parallelism.

**Part VIII — Reasoning Models**  
Chain-of-thought, test-time compute, and reasoning tokens. Strategies for spending test-time compute: best-of-N, self-consistency, reflection, verification. The cost model: reasoning tokens are decode tokens.

**Part IX — Synthesis**  
The full stack end-to-end. A benchmarking summary table: how TTFT, tokens/sec, and memory change with longer context, quantization, batching, and prefix-cache hits.

---

## Chapter Map

| Part | Chapters | Core concept |
|------|----------|-------------|
| I — Transformer Inference | 1–2 | Attention, prefill/decode, TTFT |
| II — Hardware & Systems | 3–5 | Memory bandwidth, KV cache, MoE |
| III — Optimization | 6 | Batching, caching, quantization, speculative decoding |
| IV — Sampling | 7 | Temperature, top_p, repetition penalties |
| V — Context Engineering | 8–10 | RAG, embeddings, agent context |
| VI — Evaluation | 11–12 | Offline metrics, LLM-as-Judge, RAG metrics |
| VII — Production | 13 | Serving engines, request lifecycle, parallelism |
| VIII — Reasoning | 14 | Test-time compute, chain-of-thought, cost |
| IX — Synthesis | 15 | Full stack, benchmarking summary |
| X — Labs | 16–21 | Five hands-on experiments |

---

## Hands-On Labs

Five labs turn the book into a course. Each states a hypothesis, gives starter code, and ends with what you should observe and why.

**You do not need a cluster.** A single consumer GPU — or even CPU with a small model for the sampling and RAG labs — is enough to see every effect first-hand.

```bash
pip install vllm transformers sentence-transformers qdrant-client ragas datasets
# Recommended model: Qwen2.5-7B-Instruct or Llama-3.2-3B-Instruct
```

| Lab | Reinforces | Hypothesis |
|-----|-----------|-----------|
| [Lab 1 — Measure TTFT](./lab-1/) | Parts I, VII | TTFT scales with input length; output length barely affects it |
| [Lab 2 — Watch KV Cache Growth](./lab-2/) | Part II | VRAM climbs linearly with sequence length; tokens/sec falls as cache grows |
| [Lab 3 — Compare Temperatures](./lab-3/) | Part IV | Output diversity rises with temperature; top_p only bites at high T |
| [Lab 4 — Build a RAG Pipeline](./lab-4/) | Part V | Structure and ordering improve answer quality at equal or lower token cost |
| [Lab 5 — Evaluate the RAG System](./lab-5/) | Part VI | Quality improvements show up as numbers; failure modes localize to specific stages |

---

## Key Takeaways

- **Why long contexts increase latency:** O(N²) attention in prefill, plus O(N) KV-cache bandwidth per decode step. These are different scalings — don't assume everything is quadratic.
- **How KV caching works:** computed once in prefill, read (never recomputed) in decode, grows one row per step.
- **Why batching improves throughput:** amortizes the fixed cost of loading weights from HBM across many simultaneous decode steps.
- **When quantization helps:** whenever memory is the binding constraint. INT8 for production, INT4 for edge.
- **Why context quality often beats model size:** the model can only reason over what it receives.
- **How production serves models:** queues, schedulers, continuous batching, PagedAttention, tensor and pipeline parallelism.
- **Why reasoning models cost more:** reasoning tokens are full decode tokens — time, money, and context window.

---

## What's Next

Book 1 taught you to drive the car and read the dashboard. **[Book 2](../book-2/)** takes you under the hood: GPU architecture, CUDA kernels, distributed training internals, vLLM/SGLang engine design, FlashAttention from first principles, and serving benchmarks. That progression moves from understanding inference systems to designing and optimizing them.
