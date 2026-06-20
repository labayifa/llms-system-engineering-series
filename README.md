# LLM Systems Engineering — A Three-Volume Series

**Carmel Prosper Sagbo**  
Research Associate & ML Infrastructure Engineer  
CyLab-Africa / Upanzi Network · Carnegie Mellon University Africa, Kigali

---

This repository contains the hands-on laboratory materials accompanying a three-volume technical series on LLM systems engineering. The books move from inference fundamentals to GPU internals to production infrastructure, building a continuous mental model of what actually happens from a user query to a streaming response.

## The Series

| Volume | Title | What it covers |
|--------|-------|---------------|
| **Book 1** | [LLM Inference, Sampling, and Context Engineering](./book-1/) | Attention, prefill/decode, KV cache, optimization techniques, sampling, RAG, evaluation, serving architectures, reasoning models |
| **Book 2** | [LLM Systems Engineering: CUDA, Distributed Training, and Engine Internals](./book-2/) | GPU architecture, CUDA programming model, distributed training (DDP/FSDP), vLLM/SGLang internals, FlashAttention kernels, cost modeling, serving benchmarks |
| **Book 3** | [Model Architectures, Training Techniques, and Applied Infrastructure](./book-3/) | Positional encodings (RoPE), architecture zoo, LoRA/RLHF/DPO, Kubernetes for inference, autoscaling, observability, production operations |
| **Companion** | Mathematical Appendices, References & Glossary | Derivations, annotated bibliography, cross-book glossary |

### The Arc

```
Book 1  →  attention → inference → sampling → RAG → evaluation → serving → reasoning
Book 2  →  GPU → CUDA → training → engine internals → kernels → cost & operations
Book 3  →  architectures → training methods → Kubernetes → serving platforms → production ops
```

The recurring lesson across all three volumes: **the binding constraint is almost never raw compute — it is memory, bandwidth, and communication.** KV caching, FlashAttention, quantization, PagedAttention, GQA, MoE, FSDP, and tiling are all the same move at different scales: keep data close, move it less, keep every processor fed.

---

## Repository Structure

```
llm-series/
├── README.md             ← you are here
├── book-1/               ← Inference, Sampling, Context Engineering
│   ├── README.md
│   ├── lab-1/            Lab 1 — Measure TTFT
│   ├── lab-2/            Lab 2 — Watch KV Cache Growth
│   ├── lab-3/            Lab 3 — Compare Temperatures
│   ├── lab-4/            Lab 4 — Build a RAG Pipeline
│   └── lab-5/            Lab 5 — Evaluate the RAG System
├── book-2/               ← Systems Engineering (GPU, CUDA, Kernels)
│   ├── README.md
│   ├── lab-1/            Lab 1 — Profile the Memory Hierarchy
│   ├── lab-2/            Lab 2 — Tiled vs Naive Matmul
│   ├── lab-3/            Lab 3 — Run Data-Parallel Training
│   ├── lab-4/            Lab 4 — Build a Cost Model
│   ├── lab-5/            Lab 5 — Benchmark a Server Under Load
│   └── lab-6/            Lab 6 — Measure Speculative Decoding
└── book-3/               ← Architectures, Training & Infrastructure
    ├── README.md
    ├── lab-1/            Lab 1 — Fine-Tune with LoRA
    ├── lab-2/            Lab 2 — Compare Positional Encodings
    ├── lab-3/            Lab 3 — Deploy on Kubernetes
    ├── lab-4/            Lab 4 — Autoscale Under Load
    └── lab-5/            Lab 5 — Trace a RAG Request
```

---

## Quick Start

Each lab is self-contained. The Book 1 and Book 3 (Part A) labs run on a single consumer GPU or even CPU with a small model. Book 2 kernel and distributed labs want at least one NVIDIA GPU. Book 3 Kubernetes labs run on a local cluster (kind, minikube, or k3s).

**Book 1 & 3 (Part A) setup:**
```bash
pip install vllm transformers sentence-transformers \
    qdrant-client ragas datasets peft
```

**Book 2 setup:**
```bash
pip install nvidia-ml-py torch
# NVIDIA Nsight Systems / Compute ship with the CUDA toolkit (nsys, ncu)
```

**Book 3 Kubernetes setup:**
```bash
# Install kind (local Kubernetes)
brew install kind          # macOS
kind create cluster --name llm-lab
```

A small model keeps every lab fast: `Qwen2.5-7B-Instruct` or `Llama-3.2-3B-Instruct` work well. The goal is not production scale — it is to watch the curves bend the way the theory predicts.

---

## Lab Overview

### Book 1 Labs — Inference, Sampling, Context Engineering

| Lab | Concept tested | Key observation |
|-----|---------------|-----------------|
| Lab 1 | TTFT measurement | TTFT scales with input tokens (prefill), not output tokens |
| Lab 2 | KV cache growth | VRAM rises linearly with sequence length; tokens/sec drifts down |
| Lab 3 | Temperature & top_p | Unique-output fraction rises with temperature; top_p trims the tail |
| Lab 4 | RAG construction | Context structure and ordering improve quality at equal token cost |
| Lab 5 | RAG evaluation | Ragas metrics localize failure to retrieval, ranking, or generation |

### Book 2 Labs — GPU, Kernels, Distributed Training

| Lab | Concept tested | Key observation |
|-----|---------------|-----------------|
| Lab 1 | Memory hierarchy profiling | Decode is HBM-bound; compute units sit mostly idle |
| Lab 2 | Tiled vs naive matmul | Tiling is faster despite identical FLOPs — SRAM reuse raises arithmetic intensity |
| Lab 3 | DDP / FSDP training | Near-linear throughput scaling; FSDP cuts peak memory vs DDP |
| Lab 4 | Cost modeling | Cost/token falls steeply with batch size then flattens |
| Lab 5 | Server benchmarking | p99 latency spikes before p50 at the queue-saturation knee |
| Lab 6 | Speculative decoding | Speedup tracks acceptance rate; stronger draft → higher acceptance |

### Book 3 Labs — Architectures, Training, Infrastructure

| Lab | Concept tested | Key observation |
|-----|---------------|-----------------|
| Lab 1 | LoRA fine-tuning | Trainable params drop 100×+; LoRA fits where full fine-tuning OOMs |
| Lab 2 | Positional encoding comparison | Absolute degrades past training length; RoPE extrapolates more gracefully |
| Lab 3 | Kubernetes deployment | Killed Pods respawn automatically; traffic spreads across replicas |
| Lab 4 | Autoscaling under load | Replicas track load; cold-start lag visible when scaling up |
| Lab 5 | OpenTelemetry tracing | Waterfall pinpoints latency to specific pipeline stage |

---

## About the Author

Carmel Prosper Sagbo is a Research Associate and ML Infrastructure Engineer at CyLab-Africa / Upanzi Network, Carnegie Mellon University Africa, Kigali. His work focuses on LLM inference systems, GPU infrastructure, and applied ML for African-context deployments.

GitHub: [github.com/labayifa](https://github.com/labayifa)

---

## License

The lab code in this repository is released under the MIT License. The accompanying books are released for study and reference; see individual volume notes for details.
