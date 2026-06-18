# 00 Setup: Python Dependencies

## Create a Virtual Environment

## Install Core Packages

```powershell
# NLP & Data
pip install numpy pandas matplotlib scikit-learn

# Text processing
pip install nltk spacy sentence-transformers
python -m spacy download en_core_web_sm

# HuggingFace
pip install transformers datasets accelerate

# LangChain + LangChain Community
pip install langchain langchain-community langchain-core

# LlamaIndex
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama

# Vector DB (Chroma)
pip install chromadb

# Jupyter
pip install jupyter notebook
```

## Verify Installation

```powershell
python -c "import numpy, pandas, torch, transformers, langchain, chromadb, llama_index; print('All imports OK')"
```

> **Note**: If `torch` fails, install it separately:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

## Start Jupyter

```powershell
jupyter notebook
```

All `.ipynb` files in this course can be opened and run from the Jupyter interface.
