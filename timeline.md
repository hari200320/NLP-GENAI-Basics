# 📅 Study Timeline — 8 to 10 Weeks

**Goal**: Prompt Engineer → Job-Ready GenAI Engineer

## At a Glance

| Week | Module | Hours | Type |
|------|--------|-------|------|
| 1 | 0 + 1: Setup, Python NLP Basics | 5-6 | Theory + Code |
| 2 | 1 (cont.): Embeddings, Similarity | 4-5 | Code-heavy |
| 3 | 2: Transformers, LLMs, Prompt Eng | 5-6 | Theory + Code |
| 4 | 3: LangChain | 5-6 | Code-heavy |
| 5 | 4: Vector DBs | 4-5 | Code-heavy |
| 6 | 5: LlamaIndex + Basic RAG | 5-6 | Code-heavy |
| 7 | 6: Advanced RAG | 5-6 | Code + Evaluation |
| 8 | 7: Fine-tuning (Colab) | 6-7 | Code + GPU |
| 9 | 8: Advanced Topics | 4-5 | Mixed |
| 10 | 9: Interview Prep + Review | 5-6 | Reading |

## Detailed Schedule

### Week 1: Foundations
- [ ] Read `00-setup/01-install-python-deps.md` — install everything
- [ ] Read `00-setup/02-setup-ollama.md` — pull models
- [ ] Run `01-python-nlp-basics/01-text-processing.ipynb`
- [ ] Run `01-python-nlp-basics/02-embeddings-intro.ipynb`

### Week 2: NLP & Embeddings Deep Dive
- [ ] Run `01-python-nlp-basics/03-similarity-search.ipynb`
- [ ] **Mini-project**: Build a semantic text search on 10 documents
- [ ] **Review**: Can you explain what an embedding is in 2 sentences?

### Week 3: Transformers & LLMs
- [ ] Read `02-transformers-llms/01-transformer-architecture.md`
- [ ] Run `02-transformers-llms/02-huggingface-basics.ipynb`
- [ ] Run `02-transformers-llms/03-ollama-inference.ipynb`
- [ ] Run `02-transformers-llms/04-prompt-engineering.ipynb`

### Week 4: LangChain
- [ ] Run `03-langchain/01-langchain-overview.ipynb`
- [ ] Run `03-langchain/02-chains-memory.ipynb`
- [ ] Run `03-langchain/03-agents-tools.ipynb`
- [ ] **Mini-project**: Build a local research assistant agent

### Week 5: Vector DB + LlamaIndex
- [ ] Run `04-vector-databases/01-embeddings-deep-dive.ipynb`
- [ ] Run `04-vector-databases/02-chromadb-basics.ipynb`
- [ ] Run `04-vector-databases/03-vector-search.ipynb`
- [ ] Run `05-llamaindex/01-llamaindex-basics.ipynb`
- [ ] Run `05-llamaindex/02-rag-pipeline.ipynb`

### Week 6: RAG In Depth
- [ ] Run `05-llamaindex/03-advanced-llamaindex.ipynb`
- [ ] Run `06-rag-in-depth/01-naive-rag.ipynb`
- [ ] Run `06-rag-in-depth/02-advanced-rag.ipynb`
- [ ] Run `06-rag-in-depth/03-evaluation.ipynb`
- [ ] **Mini-project**: RAG system over your own documents

### Week 7: Fine-tuning
- [ ] Read `07-fine-tuning/01-lora-qlora-theory.md`
- [ ] Open & run `07-fine-tuning/02-unsloth-finetune.ipynb` in Colab
- [ ] Run `07-fine-tuning/03-ollama-modelfile.ipynb`
- [ ] Read `07-fine-tuning/04-rlhf-dpo-intro.md`

### Week 8: Advanced Topics
- [ ] Run `08-advanced-topics/01-graph-rag.ipynb`
- [ ] Run `08-advanced-topics/02-multimodal-llms.ipynb`
- [ ] Read `08-advanced-topics/03-evaluation-metrics.md`

### Week 9-10: Interview Prep
- [ ] Read `09-interview-qa/01-fundamentals.md`
- [ ] Read `09-interview-qa/02-langchain-llamaindex.md`
- [ ] Read `09-interview-qa/03-rag-vectordb.md`
- [ ] Read `09-interview-qa/04-fine-tuning.md`
- [ ] Read `09-interview-qa/05-system-design.md`
- [ ] **Mock interviews**: Practice answering out loud

## Tips

- **Stuck?** Ask your local LLM: `ollama run llama3.2 "Explain X like I'm 5"`
- **Code not working?** Check Ollama is running: `ollama serve --foreground`
- **Want more depth?** Each notebook links to official docs
- **Portfolio**: Save your mini-projects for your GitHub/resume
