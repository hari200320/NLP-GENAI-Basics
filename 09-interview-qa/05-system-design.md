# Interview Q&A — System Design & MLOps

## System Design

**Q1: Design a RAG system for customer support.**
**A**:
```
Components:
- Document store: Product manuals, FAQs, ticket history (PDFs, HTML, DB)
- Ingestion pipeline: OCR → chunk (256-512 tokens) → embed (nomic-embed-text/all-MiniLM) → ChromaDB/Weaviate
- Query: User message → embed → search top-5 → (optional) rerank → (optional) classify intent
- Generation: System prompt + retrieved chunks + user message → LLM → response
- Memory: Conversation history (last 3 turns for context)

Considerations:
- Multi-tenant: metadata filter by customer_id/product_id
- Freshness: timestamp metadata for time-sensitive info
- Guardrails: PII detection, content moderation before/after generation
- Caching: Cache frequent queries (TTL-based)
- Fallback: If no relevant docs found → escalate or say "I don't know"
```

**Q2: How would you evaluate and monitor a production RAG system?**
**A**:
```
Online metrics:
- User feedback (thumbs up/down)
- Response time (p95 < 3s)
- Token usage (cost tracking)
- Fallback rate (no relevant docs found)

Offline metrics:
- Retrieval: Hit Rate@5, MRR
- Generation: Faithfulness, Relevance
- Regular A/B tests with golden dataset
- Regression testing on known Q&A pairs

Monitoring:
- Embedding drift (do document embeddings change over time?)
- Response length variance
- Hallucination rate (sampled human review)
- Log all queries + responses + retrieved chunks for audit
```

**Q3: Compare different approaches to scaling RAG.**
**A**:
```
Small scale (<10K docs):
- ChromaDB or FAISS
- Single process
- Simple indexing

Medium scale (10K-1M docs):
- Qdrant or Weaviate (self-hosted)
- HNSW index with tuned parameters
- Hybrid search (BM25 + dense)
- Async processing pipeline

Large scale (1M-100M+ docs):
- Pinecone or Vespa (managed)
- Multi-node sharding
- DiskANN for billion-scale
- Streaming ingestion (Kafka + vector DB connector)
- Separate indexing and query clusters
```

**Q4: How would you handle a multi-modal RAG system (text + images)?**
**A**:
1. Extract text + images from documents
2. Process separately:
   - Text → chunk → text embedding
   - Images → caption via multimodal model (LLaVA) → caption embedding
   - Images → image embedding (CLIP)
3. Store in vector DB with type metadata
4. Query: embed user query → search across text + image embeddings
5. Retrieve best result (text or image)
6. Generate final response with all retrieved info

**Q5: Design a fine-tuning pipeline for a specific domain (e.g., legal).**
**A**:
```
1. Data Collection:
   - Legal Q&A pairs, contracts, case law summaries
   - Minimum 500 high-quality examples
   - Mix domain (80%) + general (20%) to prevent forgetting

2. Data Preparation:
   - Format as instruction/response pairs
   - Clean PII/confidential info
   - Split: train (80%), validation (10%), test (10%)
   - Tokenize with chat template

3. Training:
   - QLoRA (4-bit base + LoRA rank=16)
   - Learning rate: 2e-4, cosine schedule
   - Epochs: 3 (with early stopping on eval loss)
   - Per-device batch: 2 (gradient accumulation 4)

4. Evaluation:
   - Hold-out test set (faithfulness, relevance)
   - Real legal queries reviewed by domain expert
   - Compare with base model (A/B test)

5. Deployment:
   - Merge LoRA → convert to GGUF → import to Ollama
   - Run on-premise (data privacy requirement)
   - Monitor response quality and latency
```

## MLOps

**Q6: How do you manage prompt versions?**
**A**:
- Store prompts as code (YAML/JSON files in git)
- Version control: `prompts/v1/summarizer.yaml`, `prompts/v2/summarizer.yaml`
- Include test cases with expected outputs
- Use prompt registry (LangChain Hub, custom DB)
- A/B test prompt changes with metrics
- Template variables for dynamic content

**Q7: Explain the concept of LLM observability.**
**A**: Observability = ability to understand and debug LLM behavior:
- **Tracing**: Full request flow (query → retrieved docs → prompt → response → tokens)
- **Logging**: All inputs, outputs, latency, token counts
- **Metrics**: p50/p95/p99 latency, token throughput, error rate, hallucination rate
- **Alerting**: Anomaly detection (sudden latency spike, empty responses)
- **Tools**: LangSmith, Weights & Biases, custom logging with OpenTelemetry

**Q8: How do you handle rate limiting and cost management for LLM APIs?**
**A** (if using paid APIs, but applicable for local too):
- Rate limiting: Token bucket algorithm per user/API key
- Caching: Cache identical/similar queries (semantic cache with similarity threshold)
- Queue management: Prioritize interactive queries over batch processing
- Fallback: Route to cheaper/smaller model for simple queries
- Budget tracking: Per-user/per-team token budgets with alerts
- Batching: Group similar requests for batch processing

**Q9: What is semantic caching and how would you implement it?**
**A**: Semantic caching stores query-answer pairs and checks if new queries are semantically similar to cached ones (not exact match). Implementation:
- Store queries + embeddings + responses in a vector DB
- On new query: compute embedding, search for similar cached queries
- If similarity > threshold (e.g., 0.95): return cached response
- If not: generate new response, cache it
- TTL: expire cache entries after time period

**Q10: How would you A/B test an LLM application?**
**A**:
1. Define metrics: response quality score, user satisfaction, latency, cost
2. Split traffic (50/50) between control (current) and treatment (new prompt/model)
3. Run for statistically significant sample size (typically 1-2 weeks, min 500 samples)
4. Analyze:
   - Direct metrics: latency, cost
   - Quality metrics: LLM-as-a-judge score, human eval on sample
   - User metrics: retention, engagement
5. Roll out winner or iterate if no significant difference

**Q11: How do you deploy a local LLM in production?**
**A**:
- **Ollama**: Simple, good for single-node dev/demo
- **vLLM**: Production-grade inference server, PagedAttention, continuous batching
- **TGI** (Text Generation Inference): HuggingFace's optimized server
- **llama.cpp**: CPU-friendly, GGUF format, edge deployment
- **TensorRT-LLM**: NVIDIA's optimized, max throughput on NVIDIA GPUs

Considerations: throughput (requests/sec), latency (time/token), concurrent users, GPU memory, model quantization level.

## Architecture

**Q12: Compare microservices vs. monolithic architecture for an LLM app.**
**A**:
```
Monolithic (simpler):
- Single service: ingest + query + generate
- Good for: Prototypes, <10 users, simple use cases
- Bad for: Scaling, independent deployment

Microservices (production):
- Document ingestion service
- Embedding service
- Vector DB service
- LLM inference service (vLLM cluster)
- Orchestration service (query routing, RAG pipeline)
- Evaluation/monitoring service
- Good for: Scaling, independent updates, fault isolation
- Bad for: Complexity, latency (network calls)
```

**Q13: How would you design a cost-optimized LLM inference system?**
**A**:
- Use smaller models for simple tasks (classify → BERT, generate → 3B, reason → 7B+)
- Route queries by complexity (simple → small model, complex → large model)
- Batch inference for async requests
- Quantization: INT8/FP8 for throughput, INT4 for memory
- Speculative decoding: use small draft model, verify with large model
- Prompt compression (LLMLingua, Selective Context)
- Cache frequently used prompts/responses

**Q14: How do you handle personalization in an LLM application?**
**A**:
- **In-prompt**: Inject user profile/preferences into system prompt
- **RAG**: Retrieve user-specific documents (history, preferences) as context
- **Fine-tuning**: Tune on user-specific data (if enough data per user)
- **Hybrid**: Global model + user-specific memory (vector store per user)
- **Cold start**: Use segment-based personalization (user group profiles)

**Q15: How would you ensure data privacy when using local LLMs?**
**A**:
- All processing stays on-premise (Ollama, vLLM)
- No data sent to external APIs
- PII detection before/after ingestion
- Encryption at rest (vector DB, document store)
- Role-based access control (who can query what)
- Audit logging (who queried what, when)
- Data retention policies (auto-delete old queries)
- Containerization (isolate workloads with Docker/K8s)
