# Converting Fine-Tuned Model to Ollama GGUF

After fine-tuning with Unsloth (or any LoRA adapter), you need to:
1. Merge LoRA weights into the base model
2. Convert to GGUF format (Ollama's format)
3. Create an Ollama Modelfile
4. Run it with Ollama

## Step 1: Merge LoRA Adapter with Base Model

Run this **in your Colab notebook** after training:

```python
# Merge LoRA weights into base model
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False,  # Must be False for merging
)

model.load_adapter("lora_adapter")

# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
```

## Step 2: Convert to GGUF

You need `llama.cpp` for this. In Colab:

```python
# Clone llama.cpp
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make

# Convert to GGUF (Q4_K_M quantization for best quality/size balance)
!python llama.cpp/convert.py merged_model --outfile model_q4.gguf --outtype q4_k_m
```

**Alternative**: Use `llama.cpp`'s Python bindings:

```bash
pip install llama-cpp-python
```

## Step 3: Create Ollama Modelfile

Create a file called `Modelfile`:

```dockerfile
FROM ./model_q4.gguf

# Set temperature and context length
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

# System prompt (optional)
SYSTEM """You are a helpful GenAI tutor assistant."""
```

## Step 4: Import into Ollama

```bash
# Create the model in Ollama
ollama create my-finetuned-model -f Modelfile

# Test it
ollama run my-finetuned-model "What is RAG?"
```

## Step 5: Use in Python

```python
from langchain_community.llms import Ollama

fine_tuned = Ollama(model="my-finetuned-model")
response = fine_tuned.invoke("What is LoRA?")
print(response)
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `ollama create` fails | Check GGUF file path in Modelfile |
| Model gives gibberish | Wrong quantization; try Q4_K_M |
| Too slow | Use smaller model or higher quantization |
| Can't merge adapter | Load base model without 4-bit first |

## Alternative: Direct Ollama Modelfile from HuggingFace

If you don't want to convert, you can pull from Ollama's library:

```dockerfile
# Modelfile using a base model + parameters
FROM llama3.2

# Override parameters
PARAMETER temperature 0.5
PARAMETER top_k 40

# Custom system prompt
SYSTEM "You are a specialized GenAI assistant."
```

```bash
ollama create my-custom-model -f Modelfile
```
