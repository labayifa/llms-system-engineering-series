# Book 3 — Model Architectures, Training Techniques, and Applied Infrastructure

> *From positional encodings to production platforms. The third volume completing the LLM Systems Engineering series.*

**Author:** Carmel Prosper Sagbo — CyLab-Africa / Upanzi Network, CMU Africa  
**Part of:** [LLM Systems Engineering: A Three-Volume Series](../README.md)

---

## Summary

Books 1 and 2 followed one thread: how a transformer runs, and how the hardware beneath it runs. They were deliberately narrow — inference, serving, and the systems supporting them. **Book 3 fills two gaps they left open.**

**Part A — Model Architectures and Training Techniques** fills the model-side gaps: the other architectures you encounter beyond the standard decoder, the positional-encoding designs that distinguish them, and the training methods that produce and adapt them.

**Part B — Applied LLM Infrastructure** builds the production platform: how you run a serving engine on Kubernetes, autoscale it, observe it, and orchestrate agents and RAG across many machines.

Books 1 and 2 explained one car and its engine in depth. Book 3 walks the showroom and builds the highway.

### What you learn

**Part A — Model Architectures and Training Techniques**

*Positional Encodings (Ch. 1)*  
Why transformers need position injection (attention is order-blind by construction). Sinusoidal embeddings: fixed sine/cosine waves at different frequencies, unique fingerprint per position. Learned absolute embeddings: simple, effective, capped at training length. Relative position embeddings. RoPE (Rotary Position Embedding): rotates Q and K vectors so the attention score depends only on relative position — the key to graceful context-length extrapolation.

*The Architecture Zoo (Ch. 2)*  
The sequence-model family tree: RNN, LSTM, Transformer, S4/SSM, recurrent transformer hybrids. RNN vs Transformer vs S4: the compute/memory tradeoffs of recurrence vs attention vs convolution. Notable variants (encoder-only, encoder-decoder, decoder-only). Causal vs cross-attention.

*Tokenisation and Scaling Laws (Ch. 3)*  
Byte-pair encoding and vocabulary design. Chinchilla scaling laws: optimal token/parameter ratio, and why many models are undertrained relative to compute budget.

*The Training Pipeline (Ch. 4)*  
The three stages: pretraining (next-token prediction on broad corpora), supervised fine-tuning (SFT on demonstrations), and alignment. LoRA: low-rank adaptation injects trainable rank-r matrices alongside frozen weights. Trainable parameter count drops 100×+; only adapters carry gradients and optimizer state. Target modules (q_proj, v_proj), rank r, lora_alpha scaling.

*Alignment: RLHF and Beyond (Ch. 5)*  
RLHF in three steps: collect human preference data, train a reward model, optimize the policy with PPO. The KL penalty that keeps the policy from drifting too far from the SFT base. DPO (Direct Preference Optimization): skips the reward model entirely, directly optimizes on preference pairs with a reference policy. Where reasoning models fit: RLHF with a verifiable-reward signal (math, code) produces chain-of-thought behavior without human labeling.

*Numerical and Optimization Toolkit (Ch. 6)*  
Floating-point formats (FP32, FP16, BF16, FP8) and when each is used. Exploding/vanishing gradients and gradient clipping. Fitting training into memory: gradient accumulation (simulate large batches without large memory), gradient checkpointing (recompute activations in backward to save memory at compute cost). JIT compilation with torch.compile.

**Part B — Applied LLM Infrastructure**

*Kubernetes for Inference (Ch. 7)*  
Why you need an orchestrator: self-healing, rolling updates, GPU scheduling, resource isolation. Core objects: Pod, Deployment, Service, ConfigMap, PersistentVolumeClaim. GPU scheduling: `nvidia.com/gpu` resource requests, node taints/tolerations, device plugin. HPA and KEDA for queue-depth autoscaling.

*Serving Platforms (Ch. 8)*  
The landscape: Triton Inference Server, Ray Serve, BentoML, KServe. Triton in brief: model repository, dynamic batching, gRPC/HTTP endpoints, backend plugins. Ray Serve: Python-first, composable deployments, ingress routing, graceful scaling. Multi-tenant serving: model multiplexing, LoRA adapter hot-swap, request routing by tenant.

*Autoscaling and Observability (Ch. 9)*  
Autoscaling signals: GPU utilization (misleading for decode), queue depth (most reliable), requests-per-replica. Warm-headroom strategy: keep at least one warm replica to absorb cold-start latency when scaling up. OpenTelemetry: spans, traces, metrics, logs. Trace instrumentation for multi-stage RAG and agent loops. Jaeger for waterfall visualization. Prometheus + Grafana for latency dashboards.

*Distributed RAG and Agent Orchestration (Ch. 10)*  
Sharded vector databases (millions to billions of vectors). Independent scaling of embedding, reranking, and LLM services. Ingestion pipelines running alongside serving. Agent orchestration: state management, tool execution as services, concurrency and cost control, durability via checkpointing.

*Production Incidents (Ch. 11)*  
The common failure modes mapped to their true causes: OOM on long requests (KV cache, not a leak), latency spikes under load (queue saturation, not slow model), pending GPU Pods (no schedulable node, not GPU bug), NaN losses (exploding gradients), quality regression after quantization (outlier-weight damage), rising cost per token (utilization dropped). The incident mindset: trace first, separate model from system, check cheap causes before deep ones.

---

## Chapter Map

| Part | Chapters | Core concept |
|------|----------|-------------|
| A — Architectures & Training | 1–6 | RoPE, architecture zoo, tokenization, LoRA, RLHF/DPO, numerical toolkit |
| B — Infrastructure | 7–11 | Kubernetes, serving platforms, autoscaling, observability, production ops |
| III — Labs | 12–17 | Five hands-on experiments |

---

## Hands-On Labs

Five labs close the three-book arc — two on the model side, three on the platform side.

Part A labs run on a single GPU or CPU. Part B labs want a Kubernetes cluster — a local one (kind, minikube, or k3s) is enough; you do not need a cloud GPU fleet to learn the mechanics.

```bash
# Part A
pip install peft transformers datasets torch

# Part B
brew install kind
kind create cluster --name llm-lab
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
pip install opentelemetry-sdk opentelemetry-exporter-jaeger
```

| Lab | Reinforces | Hypothesis |
|-----|-----------|-----------|
| [Lab 1 — Fine-Tune with LoRA](./lab-1/) | Part A, Ch. 4 | LoRA adapts a model on a fraction of full fine-tuning's memory |
| [Lab 2 — Compare Positional Encodings](./lab-2/) | Part A, Ch. 1 | Relative-style encodings (RoPE) extrapolate past training length; absolute do not |
| [Lab 3 — Deploy on Kubernetes](./lab-3/) | Part B, Ch. 7 | A Deployment + Service gives self-healing, load-balanced serving |
| [Lab 4 — Autoscale Under Load](./lab-4/) | Part B, Ch. 9 | Replica count tracks load when scaling on the right signal |
| [Lab 5 — Trace a RAG Request](./lab-5/) | Part B, Ch. 9–10 | A distributed trace localizes latency to a specific pipeline stage |

---

## Key Takeaways

- **RoPE enables context-length extrapolation** by encoding relative rather than absolute position in the attention score — the reason modern models handle 32k+ tokens without explicit training at that length.
- **LoRA's efficiency comes from rank:** inserting rank-r adapters into weight matrices means only 2×r×d parameters per layer carry gradients, vs d² for full fine-tuning.
- **DPO is RLHF without the reward model:** it directly optimizes on preference pairs using a closed-form loss derived from the Bradley-Terry model. Lower variance, simpler pipeline, comparable quality on many tasks.
- **Kubernetes self-heals but doesn't pre-warm:** Deployments reconcile desired state to actual state, but a new replica takes tens of seconds to load model weights. Budget warm headroom.
- **Queue depth is the right autoscaling signal for LLMs:** GPU utilization is decode-misleadingly low (memory-bound), and requests-per-second doesn't account for variable token lengths. Pending queue depth tracks actual backlog.
- **Traces attribute latency to its true source:** the incident mindset lesson — "high latency" is not a cause, it is a symptom. A trace waterfall tells you whether it is retrieval, reranking, prefill, or the queue.

---

## The Complete Arc

```
Book 1  →  attention → inference → sampling → RAG → evaluation → serving → reasoning
Book 2  →  GPU → CUDA → training → engine internals → kernels → cost & operations
Book 3  →  architectures → training methods → Kubernetes → serving platforms → production ops
```

One idea threads all three: **the binding constraint is memory, bandwidth, and communication — almost never raw compute.** It appears as the KV cache and FlashAttention in Book 1, as HBM, tiling, and FSDP in Book 2, and as gradient checkpointing, GPU scheduling, and warm-replica autoscaling in Book 3. The same move — keep data close, move it less, keep every processor fed — recurs from a single attention head to a cluster of GPU nodes.

An engineer who holds this entire arc, from a token to a production platform, and who has run the labs to feel each layer, does not merely use LLM systems. They can design, optimize, and operate them.
