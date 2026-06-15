# 00 Setup: Ollama Local LLMs

Ollama runs large language models **entirely on your machine**. No data leaves your PC.

## Step 1: Verify Ollama is Running

```powershell
ollama --version
```

If you see a version number, it's installed. If not, download from https://ollama.com

## Step 2: Pull the Models We'll Use

Open **PowerShell as Administrator** and run:

```powershell
# Main LLM for chat/inference (3B params — runs on any laptop)
ollama pull llama3.2

# Embedding model for vector search
ollama pull nomic-embed-text

# Smaller, faster model for testing
ollama pull phi3

# Multimodal model (for Module 8)
ollama pull llava
```

### Which Model When?

| Model | Size | Use Case |
|-------|------|----------|
| `llama3.2:3b` | 2.0 GB | Default chat model, RAG, agents |
| `phi3:3.8b` | 2.3 GB | Faster alternative, good for testing |
| `nomic-embed-text` | 274 MB | Convert text to embedding vectors |
| `llava:7b` | 4.5 GB | Image + text understanding |

## Step 3: Test Your Models

```powershell
ollama run llama3.2 "Explain what a vector database is in one sentence"
```

Expected output: a concise answer from the local model.

## Step 4: Ollama REST API (Used by LangChain/LlamaIndex)

Ollama runs a local API server on port 11434. Test it:

```powershell
curl http://localhost:11434/api/generate -d "{ \"model\": \"llama3.2\", \"prompt\": \"Say hello\", \"stream\": false }"
```

Python equivalent:

```python
import requests
resp = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.2",
    "prompt": "Say hello in French",
    "stream": False
})
print(resp.json()["response"])
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama` not recognized | Add Ollama to PATH or restart terminal |
| Model pulls slowly | First pull downloads ~2GB; subsequent runs are instant |
| Out of memory | Use smaller models: `llama3.2:1b` or `phi3:3.8b` |
| Port 11434 in use | `ollama serve` to restart the server |
