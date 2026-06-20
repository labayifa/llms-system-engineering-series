# Lab 4 — Build a RAG Pipeline

**Book 1 · Part V (RAG architecture, embeddings, retrieval ordering)**

---

## Hypothesis

Context structure and retrieval ordering improve answer quality at equal or lower token cost. An unstructured dump of retrieved chunks performs worse than a structured, ordered prompt, even when both contain the same retrieved content.

## Background

A RAG pipeline has two phases: **ingestion** (chunk → embed → store) and **query** (embed query → retrieve top-k → rerank → construct prompt → generate). The lost-in-the-middle effect shows that LLMs attend most strongly to content at the beginning and end of the context window — so placing the most-relevant chunks first matters. Prompt structure (Role / Task / Evidence / Format) focuses the model on the task and the retrieved evidence rather than losing it in a wall of text.

## Setup

```bash
pip install sentence-transformers qdrant-client transformers requests

# Optional: run Qdrant locally via Docker
docker run -p 6333:6333 qdrant/qdrant
```

## Document Collection

Use any small, factual document set (20–50 articles). Good options:
- Wikipedia articles on a narrow topic (e.g., African countries' capitals)
- A small PDF corpus chunked with LangChain's `RecursiveCharacterTextSplitter`
- A few dozen entries from a public QA dataset (Natural Questions, TriviaQA)

## Procedure

Build three pipeline variants and evaluate them on the same 10–15 questions:

**Variant A — Unstructured dump**  
Retrieve top-5 chunks, concatenate them in arbitrary order, and prepend them to the question with no additional structure.

**Variant B — Structured prompt**  
Retrieve top-5 chunks, format the prompt with explicit sections:
```
Role: You are a factual assistant. Answer only from the provided evidence.
Task: Answer the question below.
Evidence:
  [1] <chunk text>
  [2] <chunk text>
  ...
Question: <question>
Format: One concise paragraph.
```

**Variant C — Structured + ordered + compressed**  
Score each chunk with a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Place the highest-scoring chunk first. Drop chunks with score below a threshold to reduce token count.

For all variants, use T=0 and the same model. Record total prompt tokens, answer, and subjective correctness for each question.

## Expected Observations

- **B beats A** on most questions — structure eliminates the model's confusion about which text is the question vs the evidence.
- **C beats B**, often at *fewer* tokens — reranking and compression remove noise and place relevant content where the model attends most.
- The token cost of C is lower than A (because low-relevance chunks are dropped) even though C produces better answers.

## Why This Matters

This lab demonstrates that **context engineering often yields larger quality gains than model scaling**. The model can only reason over what it receives; a well-structured, well-ordered, compressed context gives the model the best possible input. The three-variant A/B/C framework is also a practical evaluation harness you can reuse for any RAG system.

## Variant Experiments

- Replace `all-MiniLM-L6-v2` with a stronger bi-encoder (`BAAI/bge-large-en-v1.5`) and measure retrieval precision improvement.
- Implement HyDE (Hypothetical Document Embeddings): generate a hypothetical answer to the question, embed that, and retrieve against it. Compare recall to direct query embedding.
- Add a maximum-marginal-relevance (MMR) diversification step after retrieval to reduce redundant chunks and measure its effect on context precision.
