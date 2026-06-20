# Book 2 — LLM Systems Engineering: CUDA, Distributed Training, and Engine Internals

> *The companion volume to LLM Inference, Sampling, and Context Engineering*

**Author:** Carmel Prosper Sagbo — CyLab-Africa / Upanzi Network, CMU Africa  
**Part of:** [LLM Systems Engineering: A Three-Volume Series](../README.md)

---

## Summary

Book 1 treated the GPU, kernels, and training process as black boxes. **Book 2 opens those boxes.**

This is the harder volume. It assumes you are comfortable with attention, the prefill/decode split, the KV cache, and the memory-bandwidth bottleneck from Book 1. Where a Book 1 concept is load-bearing, the text says so explicitly.

The goal: move you from understanding inference systems to being able to design and optimize them.

### What you learn

**Part I — Hardware Foundations**  
Why GPUs dominate matrix workloads: thousands of simple cores vs a few powerful CPU cores. GPU anatomy: SMs (Streaming Multiprocessors), tensor cores, SRAM (shared memory), L2 cache, HBM. How GPU interconnects (NVLink, PCIe) determine multi-GPU bandwidth. Reading spec sheets: TFLOPS, HBM bandwidth, and why bandwidth is the binding constraint for decode. The CUDA programming model: threads, blocks, grids, warps, divergence, and the memory hierarchy in code.

**Part II — Training**  
Training vs inference: two different memory profiles. The training memory breakdown — model weights, gradients, optimizer states, activations — and why training needs 4–8× more memory than inference. The forward–loss–backward–update loop. Mixed-precision training: FP16 compute with FP32 master weights, and why loss scaling matters. Distributed training strategies: data parallelism (DDP), tensor parallelism, pipeline parallelism, and fully sharded data parallelism (FSDP / ZeRO). MoE training and expert parallelism.

**Part III — Inference Engine Internals**  
vLLM deep dive: the AsyncLLMEngine, PagedAttention in detail (virtual KV pages, block tables, copy-on-write for beam search). SGLang and RadixAttention: aggressive prefix-cache reuse via a radix tree. How speculative decoding is implemented: draft model, parallel verification in the target, acceptance rate. Quantization internals: GPTQ, AWQ, SmoothQuant, the outlier-weight problem at low bit-width. KV-cache compression strategies.

**Part IV — Kernels**  
Writing a CUDA kernel from scratch: thread indexing, naive matmul, and why it is slow. Tiling: loading sub-matrices into SRAM to reuse data, raising arithmetic intensity above the roofline diagonal. Kernel fusion: fusing operations to eliminate HBM round trips. FlashAttention from first principles: the standard attention I/O problem, online softmax for numerically stable chunked computation, and how tiling avoids materializing the full N×N attention matrix.

**Part V — Operating LLM Systems**  
Cost modeling: GPU-hour → tokens/sec → cost per million tokens → cost per request. Why batching dominates unit cost (order-of-magnitude cheaper at batch 32 vs batch 1). The levers: quantization, prefix caching, speculative decoding, right-sizing the GPU. Agent system architecture: planner, tool executor, memory, verifier, and the inference-cost multiplier of multi-step trajectories. Serving benchmarks and capacity planning: the load-vs-latency curve, the saturation knee, p50 vs p99, goodput, and throughput-at-SLA.

---

## Chapter Map

| Part | Chapters | Core concept |
|------|----------|-------------|
| I — Hardware Foundations | 1–2 | GPU anatomy, CUDA model, memory hierarchy |
| II — Training | 3–5 | Training loop, mixed precision, DDP/FSDP/TP/PP |
| III — Engine Internals | 6–7 | vLLM, PagedAttention, SGLang, speculative decoding, quantization |
| IV — Kernels | 8–9 | Naive matmul, tiling, fusion, FlashAttention |
| V — Operations | 10–12 | Cost modeling, agent architecture, serving benchmarks |
| VI — Labs | 13–19 | Six hands-on experiments |

---

## Hands-On Labs

Six labs go beneath the application layer — profiling hardware, writing a kernel, running distributed, modeling cost, and benchmarking a server.

Labs 1 and 2 want a real NVIDIA GPU. Labs 4 and 5 run anywhere. Do what your hardware allows — even reading the expected results with the diagrams builds the mental model.

```bash
pip install nvidia-ml-py torch
# nsys, ncu ship with the CUDA toolkit
```

| Lab | Reinforces | Hypothesis |
|-----|-----------|-----------|
| [Lab 1 — Profile the Memory Hierarchy](./lab-1/) | Part I | Decode is HBM-bound; compute units sit mostly idle |
| [Lab 2 — Tiled vs Naive Matmul](./lab-2/) | Part IV | Tiled kernel beats naive with identical FLOPs — SRAM reuse raises arithmetic intensity |
| [Lab 3 — Run Data-Parallel Training](./lab-3/) | Part II | DDP scales throughput nearly linearly; FSDP cuts peak memory per GPU |
| [Lab 4 — Build a Cost Model](./lab-4/) | Part V | Cost per token falls steeply with batch size, then flattens |
| [Lab 5 — Benchmark a Server Under Load](./lab-5/) | Part V | Latency stays flat until the knee, then spikes; p99 degrades before p50 |
| [Lab 6 — Measure Speculative Decoding](./lab-6/) | Part III | Speedup tracks acceptance rate; stronger draft model → higher acceptance |

---

## Key Takeaways

- **The memory hierarchy dominates LLM performance:** SRAM is kilobytes-fast; HBM is gigabytes-slow. Every optimization from FlashAttention to quantization is about moving data less.
- **Decode is memory-bound, not compute-bound:** the GPU's tensor cores sit idle while waiting for HBM to deliver weights and KV cache. Arithmetic intensity sits below the roofline diagonal.
- **Tiling is the universal kernel optimization:** load a sub-matrix into SRAM, reuse it many times, write the result back once. Arithmetic intensity rises, HBM traffic falls.
- **FSDP shards vs DDP replicates:** DDP is fast but memory-heavy; FSDP is memory-light but adds communication. The right choice depends on model size vs GPU memory.
- **Batching is the dominant cost lever:** the same model on the same GPU can cost 10× more per token at batch 1 than at batch 32.
- **Capacity planning means operating below the knee:** the saturation knee in the load-vs-latency curve is where p99 starts to spike. Size your cluster so peak load hits 60–70% of the knee.
- **Speculative decoding is acceptance-rate dependent:** a draft model too slow to generate candidates negates the speedup even at high acceptance rates.

---

## The Two-Book Arc

```
Book 1  →  attention → inference → sampling → RAG → evaluation → serving → reasoning
Book 2  →  GPU → CUDA → training → engine internals → kernels → cost & operations
```

Together they form one continuous explanation. The binding constraint is almost never raw compute — it is memory, bandwidth, and communication. Attention caching, FlashAttention, quantization, PagedAttention, GQA, MoE, FSDP, and tiling are all the same move at different scales.

**[Book 3](../book-3/)** completes the series: model architectures, training methods, and the production Kubernetes platform.
