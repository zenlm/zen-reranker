---
license: apache-2.0
language:
- en
- zh
- ja
- ko
- fr
- de
- es
tags:
- reranker
- text-reranking
- semantic-search
- retrieval
- zen
pipeline_tag: text-classification
---

# Zen Reranker

**Zen Reranker** is a high-performance reranking model for search and retrieval pipelines. Part of the [Zen AI model family](https://zenlm.org) by [Hanzo AI](https://hanzo.ai).

## Overview

Zen Reranker is optimized for:
- **Retrieval-Augmented Generation (RAG)** — re-score retrieved passages for LLM context
- **Search quality improvement** — rerank initial BM25/dense retrieval results
- **Cross-lingual retrieval** — strong multilingual performance
- **DSO integration** — compatible with Hanzo's Decentralized Semantic Optimization

## Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "zenlm/zen-reranker"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16)

def rerank(query, passages):
    pairs = [[query, p] for p in passages]
    inputs = tokenizer(
        pairs, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
    ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

query = "What is the capital of France?"
passages = ["Paris is the capital of France.", "Berlin is in Germany.", "Madrid is in Spain."]
results = rerank(query, passages)
for passage, score in results:
    print(f"{score:.3f}: {passage}")
```

## With sentence-transformers

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("zenlm/zen-reranker")
scores = model.predict([
    ["What is the capital of France?", "Paris is the capital of France."],
    ["What is the capital of France?", "Berlin is in Germany."],
])
```

## Specifications

| Attribute | Value |
|-----------|-------|
| Parameters | 4B |
| Architecture | Qwen3ForSequenceClassification |
| Context | 32,768 tokens |
| Languages | 100+ (multilingual) |
| License | Apache 2.0 |

## Use Cases

1. **RAG pipelines** — rerank retrieved chunks before passing to LLM
2. **Search engines** — improve document ranking quality
3. **QA systems** — score answer candidates for relevance
4. **Semantic deduplication** — score similarity for clustering

## Abliteration

Like all Zen models, Zen Reranker is abliterated — refusal bias has been removed using directional ablation via [hanzoai/remove-refusals](https://github.com/hanzoai/remove-refusals).

**Technique**: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.

## Model Family

| Model | Parameters | Use Case |
|-------|-----------|----------|
| [Zen Nano](https://huggingface.co/zenlm/zen-nano) | 0.6B | Edge AI |
| [Zen Scribe](https://huggingface.co/zenlm/zen-scribe) | 4B | Writing |
| [Zen Pro](https://huggingface.co/zenlm/zen-pro) | 8B | Professional AI |
| [Zen Max](https://huggingface.co/zenlm/zen-max) | 671B MoE | Frontier |
| [Zen Reranker](https://huggingface.co/zenlm/zen-reranker) | 4B | Retrieval |
| [Zen Embedding](https://huggingface.co/zenlm/zen-embedding) | — | Embeddings |

## Citation

```bibtex
@misc{zen-reranker-2025,
  title={Zen Reranker: High-Performance Neural Reranking},
  author={Hanzo AI and Zoo Labs Foundation},
  year={2025},
  url={https://huggingface.co/zenlm/zen-reranker}
}
```

---
Part of the [Zen model ecosystem](https://zenlm.org) by [Hanzo AI](https://hanzo.ai) (Techstars '17) and [Zoo Labs Foundation](https://zoo.ngo).
