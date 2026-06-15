# Transformer Architecture — The Foundation of All Modern LLMs

> "Attention is All You Need" — Vaswani et al., 2017

## Why This Matters for a Prompt Engineer

Every LLM you've used (GPT, Claude, Llama, Mistral) is a **Transformer**. Understanding how it works helps you:
- Write better prompts (know what the model "sees")
- Debug unexpected outputs
- Understand why RAG works
- Communicate with engineers

## High-Level Picture

```
Input: "I love AI"
     |
     v
[Token Embedding Layer]  ← converts words to numbers
     |
     v
[Positional Encoding]    ← adds word position info
     |
     v
[Transformer Blocks]     ← 8 to 96 blocks (depending on model size)
     |   Each block contains:
     |     ├── Multi-Head Self-Attention  ← "pay attention to context"
     |     └── Feed-Forward Network       ← "process the information"
     |         (with LayerNorm + Residual connections)
     v
[Output Layer]           ← predicts next token
     |
     v
Token probabilities: "love" → 0.001, "am" → 0.9, "is" → 0.02, ...
```

## 1. Token Embedding

Words are converted to vectors using an **embedding lookup table**.

```
Vocabulary size: 50,000 tokens
Each token → 4,096-dimensional vector (for a 7B model)

"the"  → [0.12, -0.34, 0.55, ...]  (4,096 numbers)
"cat"  → [0.89, 0.12, -0.67, ...]
```

This is what we practiced in Module 1.

## 2. Positional Encoding

Transformers process all tokens in **parallel** (unlike RNNs which read sequentially).
But word order matters! Positional encoding adds this info.

```
"Dog bites man"  vs  "Man bites dog"

Each position gets a unique signal:
Position 0: [0.00, 1.00, 0.00, 1.00, ...]
Position 1: [0.84, 0.54, 0.91, -0.42, ...]
Position 2: [0.91, -0.42, 0.14, -0.99, ...]
```

## 3. Self-Attention (The Magic)

For each word, attention computes **how much to focus on every other word**.

```
Sentence: "The cat sat on the mat because it was tired"

Question: What does "it" refer to?
                                          ┌─────────┐
                                          │   it    │
                                          └────┬────┘
                                               │ Attend to:
                                    ┌──────────┼──────────┐
                                    v          v          v
                                 "The cat"   "mat"    "tired"
                                 (70%)       (20%)    (10%)
                                           ┌─┐
                                           │ │
                                           v v
                                   Answer: "cat"

So "it" gets a strong connection to "cat" — the model understands the reference.
```

### Step by step:

```
For each token, compute:
1. Query (Q): "what am I looking for?"
2. Key (K):   "what do I contain?"
3. Value (V): "my actual content"

Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Q × K^T = similarity matrix (every token with every other token)
   / √d = scale to prevent large values
softmax = turn into probabilities (0-1)
   × V  = weighted sum based on attention
```

### Multi-Head Attention

Instead of one attention calculation, models run **multiple in parallel** (8-96 heads).
Each head learns a different relationship type:

```
Head 1: "grammatical relationships" (subject-verb)
Head 2: "semantic similarity" (synonyms)
Head 3: "positional proximity" (nearby words)
Head 4: "coreference resolution" (it → cat)
...
```

## 4. Feed-Forward Network (FFN)

After attention, each token goes through a **neural network** independently:

```
Attention output → Linear Layer → ReLU/GELU → Linear Layer → Output

This is where the model "thinks" about the attended information.
Each FFN is 2-4x the embedding dimension (e.g., 4096 → 11008 → 4096)
```

## 5. Layer Normalization + Residual Connections

```
     Input
       │
       v
 [LayerNorm]      ← normalize values (stable training)
       │
       v
 [Attention]      ← compute relationships
       │
       v
   + (add input)  ← residual connection (preserves original info)
       │
       v
 [LayerNorm]
       │
       v
    [FFN]
       │
       v
   + (add input)  ← another residual
       │
       v
     Output
```

**Residual connections** let gradients flow through deep networks (enables 100+ layers).

## 6. Causal Masking (Decoder-Only Models)

Models like GPT, Llama, Mistral are **decoder-only**:
They can only attend to **previous tokens** (not future ones).

```
"I love" → predict next token

Position 0 ("I"): can only see "I"
Position 1 ("love"): can see "I", "love"
Position 2 (?): can see "I", "love", ?

Mask:
[1, 0, 0]   ← token 0 sees only itself
[1, 1, 0]   ← token 1 sees tokens 0,1
[1, 1, 1]   ← token 2 sees all previous
   ↑ attention matrix
```

This is why LLMs are **autoregressive** — they predict one token at a time.

## 7. Training vs. Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Goal** | Learn parameters | Generate text |
| **Data** | Billions of tokens | User prompt |
| **Process** | Predict next token, backpropagate error | Predict next token, feed it back in |
| **Parallel** | Yes (teacher forcing) | No (sequential) |
| **Cost** | Millions of $ | Cents per query |

## 8. Model Sizing

| Model | Parameters | Layers | Heads | Hidden Dim | VRAM |
|-------|-----------|--------|-------|-----------|------|
| Llama 3.2 (1B) | 1B | 16 | 32 | 2048 | ~2 GB |
| Llama 3.2 (3B) | 3B | 28 | 24 | 3200 | ~6 GB |
| Llama 3 (8B) | 8B | 32 | 32 | 4096 | ~16 GB |
| GPT-4 | ~1.8T (est.) | ~120 | ~96 | ~12288 | $$$ |

## Your Turn

Run the next notebook (`02-huggingface-basics.ipynb`) to load a real transformer model and see these concepts in action.
