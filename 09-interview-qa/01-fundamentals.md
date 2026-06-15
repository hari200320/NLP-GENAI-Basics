# Interview Q&A — Fundamentals

## Core Concepts

**Q1: What is a Transformer and why is it important?**
**A**: A Transformer is a neural network architecture introduced in "Attention is All You Need" (2017). It uses **self-attention** to process all tokens in parallel, unlike RNNs which process sequentially. This parallelization enabled training on massive datasets, leading to modern LLMs like GPT, Llama, and Mistral. Its key innovations are multi-head attention, positional encoding, and the encoder-decoder structure (though modern LLMs use decoder-only variants).

**Q2: Explain self-attention in simple terms.**
**A**: Self-attention allows each token to "look at" every other token and decide how much to focus on each. For each token, we compute Query (what I'm looking for), Key (what I contain), and Value (my content) vectors. The attention score = softmax(Q×K^T/√d) tells each token how much to weigh every other token. This captures relationships like "it" → "cat" in a sentence.

**Q3: What's the difference between encoder-only, decoder-only, and encoder-decoder models?**
**A**:
- **Encoder-only** (BERT): Bidirectional context. Best for understanding tasks (classification, NER).
- **Decoder-only** (GPT, Llama): Left-to-right causal attention. Best for generation tasks.
- **Encoder-Decoder** (T5, BART): Full transformer stack. Best for seq2seq tasks (translation, summarization).

**Q4: What is tokenization and why does it matter?**
**A**: Tokenization converts text into numbers (tokens). Modern LLMs use **subword tokenizers** (BPE, SentencePiece, WordPiece). Key points:
- "unhappiness" → ["un", "happi", "ness"]
- Token count ≠ word count (important for context windows)
- Different models use different tokenizers (affects how prompts are processed)
- ~100 tokens ≈ 75 English words

**Q5: What is the difference between a model's context window and its training data?**
**A**: The **context window** is the maximum input length the model can process at inference time (e.g., 128K tokens for GPT-4, 8K for Llama 2). The model "sees" everything in the window. **Training data** is the billions of documents the model learned from during pre-training. The model's knowledge comes from training data; the context window provides in-session information.

**Q6: What is temperature in LLMs?**
**A**: Temperature controls the randomness of token selection. It scales the logits before softmax:
- Low (0.1): Always picks the highest probability token (deterministic, repetitive)
- Medium (0.7): Balanced creativity (default)
- High (1.5): Near-random selection (creative, may be incoherent)
- Temperature = 0: Always same output (greedy decoding)

**Q7: What is top-k and top-p sampling?**
**A**: Alternative strategies to temperature:
- **Top-k**: Only sample from the K highest probability tokens (e.g., top-50)
- **Top-p** (nucleus): Sample from the smallest set of tokens whose cumulative probability ≥ p (e.g., p=0.9)
Both are often used together with temperature for better control.

**Q8: What's the difference between fine-tuning and training from scratch?**
**A**: **Training from scratch** initializes random weights and trains on massive data (costs millions, needs thousands of GPUs). **Fine-tuning** starts from a pretrained model and continues training on a smaller, domain-specific dataset. Fine-tuning is orders of magnitude cheaper and requires much less data.

## LLM Concepts

**Q9: What is hallucination in LLMs?**
**A**: Hallucination is when an LLM generates information that's factually incorrect or not grounded in its input. Causes:
- Model learned patterns, not facts
- Training data had errors
- Model tries to be helpful even when uncertain
- No retrieval mechanism to verify claims

**Q10: How does RAG reduce hallucinations?**
**A**: RAG provides **relevant context documents** to the LLM before generation. The LLM is instructed to answer based only on that context. This grounds the output in retrieved information rather than relying solely on the model's parametric memory.

**Q11: What is the difference between pre-training and fine-tuning data?**
**A**: **Pre-training data** is massive (trillions of tokens), general domain (web crawl, books, Wikipedia). The model learns language, grammar, reasoning, and broad knowledge. **Fine-tuning data** is small (hundreds to thousands of examples), task-specific (Q&A pairs, instructions, preference pairs). The model adapts its behavior/style to the specific use case.

## Technical

**Q12: What are logits?**
**A**: Logits are the raw, unnormalized scores the model outputs for each token in its vocabulary. They go through softmax to become probabilities. Logits can be positive or negative, and their relative values determine which tokens are most likely.

**Q13: Explain the difference between inference and training in terms of compute.**
**A**: **Training**: Forward pass (compute predictions) + backward pass (compute gradients) + optimizer step (update weights). Requires storing activations for all layers (~3× model size for Adam). **Inference**: Forward pass only. Much cheaper. For a 7B model: training needs ~56GB VRAM, inference needs ~14GB (16-bit).

**Q14: What is KV caching?**
**A**: During autoregressive generation, each new token attends to all previous tokens. Instead of recomputing Key and Value vectors for every previous token at each step, **KV cache** stores them. This trades memory for speed. A 7B model with 4K context uses ~2GB for KV cache.

**Q15: What's the difference between greedy decoding and beam search?**
**A**: **Greedy**: picks the highest probability token at each step (fast but may miss better sequences). **Beam search**: keeps the top-B candidate sequences (beams) at each step, exploring multiple paths. B=4 is common. Beam search finds more optimal sequences but is B× slower.
