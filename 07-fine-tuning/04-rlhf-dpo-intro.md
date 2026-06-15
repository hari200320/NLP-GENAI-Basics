# RLHF & DPO — Preference Optimization

## The Problem

LLMs are trained on **next-token prediction** — they learn what's *likely*, not what's *helpful*.

```
Training data:  "The answer is 42." ✓ (correct, factual)
               "The answer is purple." ✓ (also a valid sentence)

The model learns both. It doesn't know which is *better*.
```

**Solution**: Align model outputs with human preferences.

## RLHF (Reinforcement Learning from Human Feedback)

The original alignment technique (used by ChatGPT/GPT-4).

### 3-Step Process

```
Step 1: Collect human preferences
Human: Which response is better?
  A: "I don't know the answer."
  B: "Here's a detailed explanation with sources..."
Human prefers B.

Step 2: Train a Reward Model
  - Binary classifier: predict which response humans prefer
  - Scores responses: "good" vs "bad"

Step 3: Fine-tune LLM with RL
  - Use PPO (Proximal Policy Optimization)
  - Reward model gives feedback: "make outputs like B, not like A"
  - KL penalty prevents model from drifting too far
```

**Problem**: RLHF is complex, expensive, and unstable. Need to maintain 4 models simultaneously (policy, reference, reward, critic).

## DPO (Direct Preference Optimization) — The Modern Alternative

DPO simplifies RLHF by **reformulating the objective**.

### How DPO Works

Instead of:
1. Train reward model → 2. Run RL with reward

DPO directly optimizes the policy using preference pairs:

```
Given: preference pairs (chosen_response, rejected_response)

DPO loss = -E[log σ(β * (reward_chosen - reward_rejected))]

Where:
- σ = sigmoid function
- β = temperature parameter
- reward = implicit reward derived from policy ratio
```

**Key insight**: The optimal reward function can be expressed **directly in terms of the policy**. No separate reward model needed!

### DPO vs RLHF

| Aspect | RLHF | DPO |
|---|---|---|
| Complexity | High (4 models) | Low (1 model) |
| Stability | Unstable | Stable |
| Data efficiency | Low | High |
| Implementation | Hard | Easy |
| Popularity | Decreasing | Increasing |

## Preference Datasets

Format for DPO:

```json
[
  {
    "prompt": "What is RAG?",
    "chosen": "RAG is Retrieval Augmented Generation...",
    "rejected": "RAG is a type of carpet."
  },
  {
    "prompt": "Explain transformers",
    "chosen": "Transformers use self-attention...",
    "rejected": "Transformers are robots in disguise."
  }
]
```

## When to Use Which

| Goal | Approach |
|---|---|
| Teach new knowledge | Fine-tuning (LoRA) |
| Teach format/style | Fine-tuning (LoRA) |
| Make model helpful/honest | DPO between good/bad examples |
| Remove harmful outputs | DPO on safety data |
| Maximum alignment | RLHF (if you have the budget) |

## Practical DPO with Unsloth

```python
from unsloth import FastLanguageModel
from trl import DPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    load_in_4bit=True,
)

# Load preference dataset
dataset = load_dataset("your_preference_data")

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # DPO uses implicit reference
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-6,
    ),
    beta=0.1,  # DPO temperature parameter
)

trainer.train()
```

## Key Takeaways

1. **RLHF**: 3-stage pipeline (collect preferences → train reward model → RL fine-tuning)
2. **DPO**: Simplified alternative — no reward model needed
3. DPO is replacing RLHF as the default alignment method
4. Both require preference data (chosen vs rejected responses)
5. For most use cases, **LoRA + DPO** is the sweet spot
