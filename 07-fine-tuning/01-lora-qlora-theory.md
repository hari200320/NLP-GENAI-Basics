# LoRA, QLoRA & PEFT — Theory

## Why Fine-Tune?

| Approach | When | Cost |
|---|---|---|
| **Prompt Engineering** | Simple tasks, quick prototype | Free |
| **RAG** | Need up-to-date / external knowledge | Free |
| **Fine-tuning** | Need specific behavior / style / format | Requires GPU |

Fine-tuning = continuing training on domain data to **adapt the model's behavior**.

## Full Fine-Tuning

Update **all** parameters of the model.

```
Original model: 7B parameters
Full fine-tune: 7B parameters updated
VRAM needed: ~56 GB (7B × 8 bytes for Adam)
```

**Problem**: Too expensive for most people. Need multiple GPUs.

## PEFT (Parameter-Efficient Fine-Tuning)

Only train a small subset of parameters while freezing the rest.

### LoRA (Low-Rank Adaptation) — The Most Popular

**Key insight**: Weight updates during fine-tuning have a **low intrinsic rank** (they lie in a low-dimensional subspace).

Instead of updating a full weight matrix `W` (size `d × k`), LoRA adds two small matrices `A` and `B`:

```
Original:  W (d × k) — frozen
LoRA:      B × A      — trainable
              ↑   ↑
          d × r  r × k    where r << d, k (e.g., r=8, d=4096)

Trainable params = d×r + r×k   vs   d×k   (typically 99% fewer!)
```

**Diagram**:

```
    ┌──────────────┐
    │   Input x    │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  Frozen W    │  ← original weights (not updated)
    │  (d × k)     │
    └──────┬───────┘
           │           ┌──────┐
           │    +      │ B×A  │  ← LoRA adapter (trained)
           │           │ d×r  │
           │           └──────┘
           │
    ┌──────┴───────┐
    │   Output y   │
    └──────────────┘
```

**Key parameters**:
- `r` (rank): 4, 8, 16, 32, 64. Higher = more capacity, more VRAM.
- `alpha`: scaling factor (usually 2× rank)
- `target_modules`: which layers to apply LoRA to (usually attention layers)

### QLoRA (Quantized LoRA)

QLoRA = LoRA + 4-bit quantization of the base model.

```
Base model:  4-bit NormalFloat (NF4)  → ~4× less memory
LoRA:        still 16-bit (BF16)
Combined:    ~4GB for a 7B model (vs ~56GB for full FT)
```

**Memory comparison** (7B model):

| Method | VRAM Needed | Quality |
|---|---|---|
| Full fine-tune | ~56 GB | Best |
| LoRA (16-bit) | ~16 GB | ~98% of full |
| QLoRA (4-bit) | ~4-6 GB | ~97% of full |
| QLoRA (8-bit) | ~8 GB | ~98% of full |

**What you can run on which GPU**:

| GPU | VRAM | Can train |
|---|---|---|
| Colab T4 (free) | 16 GB | QLoRA 7B, LoRA 1-3B |
| RTX 3060 | 12 GB | QLoRA 7B |
| RTX 4090 | 24 GB | QLoRA 13B, LoRA 7B |
| A100 (cloud) | 80 GB | Full FT 7B, QLoRA 70B |

## Training Process

```
1. Load base model (quantized if QLoRA)
2. Add LoRA adapters to target modules
3. Prepare dataset (instruction format)
4. Train with low learning rate (1e-4 to 5e-4)
5. Save only the LoRA adapter weights (small!)
6. Merge adapter with base model (optional)
7. Export to GGUF for Ollama (if desired)
```

## Step-by-Step Colab Workflow

The next notebook (`02-unsloth-finetune.ipynb`) is designed for **Google Colab's free T4 GPU**.

You will:
1. Open the notebook in Colab (File → Upload Notebook)
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (takes ~15-30 min)
4. Download the fine-tuned LoRA adapter

## Common Fine-Tuning Mistakes

| Mistake | Fix |
|---|---|
| Learning rate too high | Use 2e-4 for LoRA, 1e-4 for QLoRA |
| Dataset too small | Need at least 100-1000 examples |
| Overfitting | Use eval set, early stopping |
| Wrong chat template | Match the model's expected format |
| Training too long | 1-3 epochs is usually enough |

**Pro tip**: Always test your fine-tuned model on held-out examples before deploying.
