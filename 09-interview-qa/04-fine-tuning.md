# Interview Q&A — Fine-Tuning

## Concepts

**Q1: When should you fine-tune vs. use RAG vs. prompt engineer?**
**A**:
- **Prompt engineering**: Simple tasks, quick prototype, no cost. Most accessible.
- **RAG**: Need external/up-to-date knowledge, private documents. No training needed.
- **Fine-tuning**: Need specific behavior/style/format, task is hard to prompt. Requires GPU + data.
- **Both RAG + fine-tuning**: Production systems often use both: RAG for knowledge, fine-tuning for behavior.

**Q2: What is the difference between full fine-tuning and PEFT?**
**A**: **Full fine-tuning** updates all model parameters (billions of weights). High quality, but needs massive GPU memory (56GB+ for 7B). **PEFT** (Parameter-Efficient Fine-Tuning) updates a tiny fraction of parameters while freezing the rest. Lower memory, faster training, ~1-3% quality loss vs full. LoRA is the most popular PEFT method.

**Q3: Explain LoRA in detail.**
**A**: LoRA (Low-Rank Adaptation) is based on the insight that weight updates during fine-tuning have low intrinsic rank. Instead of updating `W` (d×k), it adds two small matrices `B` (d×r) and `A` (r×k) where r << d,k. The forward pass becomes `h = Wx + BAx`. Typically r=8-64, reducing trainable parameters by ~99%. LoRA is applied to attention projection matrices (Q, K, V, O).

**Q4: What is QLoRA and how does it work?**
**A**: QLoRA = Quantized LoRA. It quantizes the base model to 4-bit (NF4 format) while keeping LoRA adapters in 16-bit. This reduces memory ~4x: a 7B model can be fine-tuned on 6GB VRAM (vs 24GB for regular LoRA). The key innovations are: 4-bit NormalFloat quantization, double quantization, and paged optimizers for memory spikes.

**Q5: What are the key hyperparameters in LoRA?**
**A**:
- `r` (rank): 4-64. Higher = more capacity but more parameters. Start with 8 or 16.
- `lora_alpha`: scaling factor (typically 2× r). Higher = stronger adapter effect.
- `lora_dropout`: 0 usually (modern implementations). 0.1 for regularization.
- `target_modules`: which layers to apply LoRA to. Typically Q, K, V, O projections.
- `bias`: 'none' (default), 'all', or 'lora_only'.

**Q6: How do you choose the right learning rate for fine-tuning?**
**A**: 
- Full fine-tuning: 1e-5 to 5e-5
- LoRA: 1e-4 to 5e-4 (higher because fewer parameters are updated)
- QLoRA: 1e-4 to 3e-4
- DPO: 5e-6 to 1e-5 (much lower, preference optimization is delicate)
Best practice: cosine scheduler with warmup (5-10% of steps).

**Q7: What dataset format do you need for instruction fine-tuning?**
**A**: Common formats:
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}` (instruction + optional input + response)
- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` (multi-turn)
- **ChatML**: Uses special tokens for role-based conversations
The format must match the model's training template (e.g., Llama's chat template).

**Q8: How much data do you need for fine-tuning?**
**A**:
- **100-500 examples**: Noticeable behavior change (style, format)
- **500-5,000 examples**: Good task adaptation
- **5,000-50,000+**: Best quality (but diminishing returns)
- Quality > quantity: 1,000 diverse, high-quality examples > 10,000 noisy ones

## Practical

**Q9: What hardware do you need for fine-tuning?**
**A** (assuming 7B model):
- **Full fine-tune**: 8× A100 80GB (≈$40/hr cloud)
- **LoRA (16-bit)**: 1× RTX 3090/4090 (24GB VRAM)
- **QLoRA (4-bit)**: 1× T4 (16GB, free Colab) or RTX 3060 (12GB)
- **LoRA (8-bit)**: 1× RTX 3060 (12GB)
For 3B models (like Llama 3.2): half the VRAM requirements.

**Q10: Explain the training loop for fine-tuning.**
**A**:
1. Load base model (optionally quantized)
2. Add LoRA adapters
3. Load and tokenize dataset
4. Initialize SFTTrainer (or custom trainer)
5. Train for 1-3 epochs
6. Save LoRA adapter weights (small file, ~16MB for 7B)
7. Optional: merge adapter with base model
8. Evaluate on held-out test set

**Q11: What is catastrophic forgetting and how do you prevent it?**
**A**: Catastrophic forgetting happens when fine-tuning on new data causes the model to lose previously learned capabilities. Prevention:
- Mix some general domain data with your fine-tuning data (5-10%)
- Use lower learning rates
- Train for fewer epochs
- Use LoRA (freezing most weights preserves original knowledge)
- Use replay buffers (replay old tasks)

**Q12: What is DPO and how is it different from RLHF?**
**A**: DPO (Direct Preference Optimization) reformulates RLHF to work without a separate reward model. Instead of: train reward model → RL optimize with PPO, DPO directly optimizes the policy using preference pairs (chosen vs rejected response). It's simpler, more stable, and 3× cheaper than RLHF.

**Q13: What is overfitting in fine-tuning?**
**A**: Overfitting = model memorizes training examples but can't generalize. Signs: training loss goes to 0, eval loss increases, model repeats training examples verbatim. Solutions: more data, early stopping, higher dropout, lower rank (r), data augmentation, eval set monitoring.

**Q14: How do you convert a fine-tuned model to GGUF for Ollama?**
**A**: 1. Merge LoRA adapter with base model → 2. Convert to GGUF using llama.cpp's convert.py → 3. Create Modelfile pointing to GGUF → 4. `ollama create mymodel -f Modelfile`. The GGUF format uses 4-bit quantization (Q4_K_M recommended) for efficient local inference.

**Q15: What's the difference between supervised fine-tuning (SFT) and preference optimization (DPO)?**
**A**: **SFT** teaches the model *what to say* (demonstration data: input → desired output). **DPO** teaches the model *what's better* (preference data: input → chosen vs rejected output). SFT is for task learning, DPO is for alignment. They're often combined: SFT first, then DPO.
