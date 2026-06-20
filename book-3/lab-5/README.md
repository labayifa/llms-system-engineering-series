# Lab 5 — Trace a RAG Request

**Book 3 · Part B, Chapters 9–10 (observability, distributed RAG)**

---

## Hypothesis

A distributed trace with one span per pipeline stage localizes latency to its true source. When one stage is deliberately slowed, the trace waterfall immediately identifies it — without needing to reproduce the slowdown in isolation or rely on aggregate metrics.

## Background

Distributed tracing propagates a `trace_id` through a multi-service request. Each service creates a **span** (start time, end time, service name, attributes) as a child of the incoming trace context. The resulting **waterfall diagram** shows all spans on a shared time axis, making it trivial to see which stage took how long. In a RAG pipeline (embed → retrieve → rerank → generate), each stage has a distinct latency profile: embedding is fast (milliseconds), retrieval depends on index size, reranking is a small model forward pass, and generation is the dominant cost. Traces make the relative contributions visible and attributable.

OpenTelemetry is the open standard for instrumentation. Jaeger is a common open-source trace backend that provides the waterfall UI.

## Setup

```bash
# Python OpenTelemetry SDK and Jaeger exporter
pip install opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc \
            opentelemetry-instrumentation-requests

# Run Jaeger locally
docker run -d -p 16686:16686 -p 4317:4317 \
    jaegertracing/all-in-one:latest

# Jaeger UI will be available at http://localhost:16686
```

## Procedure

### Part A — Instrument the RAG pipeline
Wrap each pipeline stage in an OpenTelemetry span. At a minimum, create spans for:
- `embed_query` — time to embed the user's question
- `retrieve` — time to query the vector database
- `rerank` — time to run the cross-encoder (if present)
- `generate` — time for the LLM to produce the answer

```python
from opentelemetry import trace

tracer = trace.get_tracer("rag-pipeline")

with tracer.start_as_current_span("embed_query"):
    query_embedding = embedder.encode(question)

with tracer.start_as_current_span("retrieve"):
    chunks = vector_db.search(query_embedding, top_k=5)

with tracer.start_as_current_span("rerank"):
    ranked = reranker.rank(question, chunks)

with tracer.start_as_current_span("generate"):
    answer = llm.generate(build_prompt(question, ranked))
```

Add span attributes for relevant metadata: number of chunks retrieved, token counts, model names.

### Part B — Inspect normal trace
1. Send 10 requests and view the traces in Jaeger.
2. Note the typical duration of each stage and their relative proportions.

### Part C — Inject a slowdown and confirm trace attribution
1. Add an artificial `time.sleep(2)` inside the `rerank` span.
2. Send another 5 requests.
3. Open Jaeger and confirm the `rerank` span is visually dominant in the waterfall.
4. Remove the sleep and confirm the trace returns to its normal profile.

## Expected Observations

- **Normal traces:** `generate` dominates (LLM inference is the slowest stage). `retrieve` is typically next. `embed_query` and `rerank` are fast.
- **With injected slowdown:** the `rerank` span widens to dominate the waterfall — immediately visible without any additional debugging.
- **Trace consistency:** across 10 requests, the relative stage durations should be stable. Variability in `generate` latency (from batching and queue dynamics) should also be visible.
- **Attributes enrich the trace:** filtering by chunk count or model name in Jaeger lets you correlate performance with pipeline configuration.

## Why This Matters

This lab is the observability counterpart to Lab 5 in Book 1 (RAG evaluation). Ragas tells you *what* is wrong with quality. Traces tell you *where* the latency went. Together they cover both dimensions of production RAG health. The incident mindset from Book 3's Chapter 11 — "look at the trace first; separate model from system" — is directly practiced here: the deliberate slowdown makes it unambiguous that the problem is reranking, not the LLM, even though p99 TTFT rose equally for both.

## Variant Experiments

- Propagate the trace across service boundaries: deploy the embedder, reranker, and LLM as separate HTTP services, and use the W3C `traceparent` header to connect the spans across services in Jaeger.
- Add a `vllm:num_requests_waiting` gauge as a span attribute on the `generate` span — correlate queue depth at request time with generate latency in the trace.
- Set up a Grafana dashboard that plots p50 and p99 of each span duration over time — this is the permanent observability layer, complementing the per-request waterfall view in Jaeger.
