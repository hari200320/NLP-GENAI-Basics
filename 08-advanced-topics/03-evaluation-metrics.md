# Evaluation Metrics for NLP & GenAI

## 1. Intrinsic Metrics (How good is the model itself?)

### Perplexity
**What**: Measures how "surprised" the model is by test data.
**Lower = better**.

```
Perplexity = exp(-1/N * sum(log P(token_i | context)))

GPT-2: ~35 perplexity on WikiText-2
GPT-3: ~20
Llama 3.2 (3B): ~12
```

**Use**: Comparing language models on the same dataset.

### Cross-Entropy Loss
**What**: The training loss — how wrong is the next-token prediction.

### Accuracy
**What**: For classification tasks (sentiment, NER).

## 2. Generation Metrics (How good is the generated text?)

### BLEU (Bilingual Evaluation Understudy)
**What**: N-gram overlap between generated and reference text.

```
BLEU = precision of n-grams with brevity penalty

Score range: 0-100
Human translation: ~40 BLEU
Good MT: ~30-40
```

**Limitation**: Punishes valid paraphrases. Not great for creative text.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
**What**: Recall of n-grams (vs BLEU's precision).

```
ROUGE-1: unigram overlap
ROUGE-2: bigram overlap
ROUGE-L: longest common subsequence
```

**Use**: Summarization evaluation.

### METEOR
**What**: Improves on BLEU by using stemming and synonyms.

### BERTScore
**What**: Uses BERT embeddings to compute semantic similarity between generated and reference text.

```
BERTScore = cosine similarity of contextual embeddings

Range: 0-1 (≈0.9 for good summaries)
```

**Use**: More robust than BLEU/ROUGE for semantic similarity.

## 3. LLM-Specific Metrics

### LLM-as-a-Judge
**What**: Use a strong LLM (GPT-4, Claude) to rate outputs.

```
Prompt: "Rate this response on 1-5 for helpfulness: [response]"
```

**Pros**: Flexible, catches semantic issues
**Cons**: Expensive, biased toward the judge model

### MT-Bench / Chatbot Arena
**What**: Multi-turn conversation evaluation.
- MT-Bench: 80 multi-turn questions, scored by LLM
- Chatbot Arena: Head-to-head human voting

### HELM (Holistic Evaluation of Language Models)
**What**: Comprehensive benchmark suite:
- 7 scenarios (QA, summarization, etc.)
- 7 metrics (accuracy, calibration, bias, etc.)
- 42+ model comparisons

## 4. RAG-Specific Metrics

| Metric | What It Measures | Formula |
|---|---|---|
| **Hit Rate** | Does relevant doc appear in top-k? | Relevant docs in top-k / total relevant |
| **Mean Reciprocal Rank (MRR)** | How early does the first relevant doc appear? | 1/rank_first_relevant (averaged) |
| **Normalized Discounted Cumulative Gain (NDCG)** | Ranking quality with graded relevance | DCG / ideal DCG |
| **Faithfulness** | Are LLM claims supported by context? | LLM judges each claim |
| **Answer Relevance** | Does answer address the question? | LLM or cosine similarity |

**Implementing Hit Rate**:

```python
def hit_rate(retrieved_docs, relevant_doc_ids, k=3):
    """Check if any relevant doc is in top-k retrieved."""
    hits = 0
    for query_docs, relevant in zip(retrieved_docs, relevant_doc_ids):
        if any(doc_id in relevant for doc_id in query_docs[:k]):
            hits += 1
    return hits / len(retrieved_docs)
```

## 5. When to Use What

| Task | Primary Metric | Secondary |
|---|---|---|
| Language modeling | Perplexity | Cross-entropy loss |
| Translation | BLEU | BERTScore |
| Summarization | ROUGE | BERTScore |
| RAG generation | Faithfulness | Answer Relevance |
| RAG retrieval | Hit Rate@k | MRR, NDCG |
| Chatbot quality | MT-Bench | LLM-as-a-Judge |
| Instruction following | AlpacaEval | Human eval |

## 6. Practical Advice

1. **Don't rely on a single metric** — each has blind spots
2. **Human evaluation** is still the gold standard (but expensive)
3. **For RAG**, measure retrieval and generation separately
4. **LLM-as-a-Judge** is cheap but biased — use multiple judges
5. **Always evaluate on your specific task**, not just general benchmarks
6. **Track metrics over time** to catch regressions
