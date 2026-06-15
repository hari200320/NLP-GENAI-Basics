# Interview Q&A — RAG & Vector Databases

## RAG (Retrieval Augmented Generation)

**Q1: What is RAG and why is it needed?**
**A**: RAG = Retrieval Augmented Generation. It combines a retrieval system with an LLM to generate grounded answers. It's needed because:
1. LLMs have limited knowledge (training cutoff)
2. LLMs hallucinate when uncertain
3. LLMs can't access private/enterprise data
4. Fine-tuning is expensive for knowledge updates

**Q2: Explain the RAG pipeline step by step.**
**A**:
1. **Ingestion**: Documents → chunk → embed → store in vector DB
2. **Query**: User asks question → embed query → search vector DB
3. **Retrieval**: Get top-K relevant chunks (with metadata filtering)
4. **Augmentation**: Concatenate retrieved chunks as context
5. **Generation**: LLM generates answer based on context + question

**Q3: What are the failure modes of RAG?**
**A**:
- **Missing content**: No relevant docs retrieved (increase K, improve embeddings)
- **Missed ranking**: Relevant doc exists but not in top-K (reranking, HyDE)
- **Not in context**: Retrieved but LLM ignores it (prompt engineering)
- **Wrong format**: Retrieved docs don't match answer format (chunk at right level)
- **Incorrect specificity**: Retrieved too broad/narrow (adjust chunk size)

**Q4: How do you evaluate RAG quality?**
**A**: Two aspects:
- **Retrieval**: Hit Rate@K, MRR, NDCG
- **Generation**: Faithfulness (claims match context), Answer Relevance (addresses question), Answer Correctness (factual accuracy)

**Q5: What is HyDE and how does it improve RAG?**
**A**: HyDE (Hypothetical Document Embeddings) generates a hypothetical answer first, then uses that answer's embedding (instead of the query embedding) for retrieval. This bridges the "vocabulary gap" between short queries and full document texts. The hypothetical document looks more like the target documents than the query does.

**Q6: What is multi-query retrieval?**
**A**: The LLM generates multiple variations of the user's question, searches with each variation, merges results, and deduplicates. This improves recall by covering different phrasings of the same information need.

**Q7: What is reranking and when should you use it?**
**A**: Reranking uses a more accurate (but slower) model to re-score the initially retrieved documents. Typically: retrieve top-20 with embedding similarity → rerank with cross-encoder → return top-3. Use when you need higher precision and have the compute budget for a cross-encoder.

**Q8: What is the difference between sparse and dense retrieval?**
**A**: **Sparse** (BM25, TF-IDF): keyword-based, fast, good for exact match, no training needed. **Dense** (embedding-based): semantic search, needs training, slower but captures meaning. **Hybrid search** combines both for best results.

**Q9: How do you handle document updates in a RAG system?**
**A**: 
- Re-embed and re-insert updated documents (with same ID to overwrite)
- Use metadata timestamps for filtering
- Incremental indexing for high-frequency updates
- Consider change data capture (CDC) patterns for databases

**Q10: What is "lost in the middle" in RAG?**
**A**: LLMs tend to focus on information at the beginning and end of long contexts, ignoring the middle. To mitigate: place the most relevant documents first or last, use structured formatting, or reduce context length.

## Vector Databases

**Q11: What is a vector database and how is it different from a regular database?**
**A**: A vector database stores **embeddings** (vectors) and enables **similarity search** using distance metrics (cosine, L2, dot product). Regular databases use exact matching with indexes (B-tree). Vector DBs use ANN (Approximate Nearest Neighbor) algorithms for fuzzy/semantic matching.

**Q12: What ANN algorithms are commonly used?**
**A**:
- **HNSW** (Hierarchical Navigable Small World): Graph-based, fast search, high memory
- **IVF** (Inverted File Index): Cluster-based, less memory, slower
- **IVF-PQ**: IVF with Product Quantization, even less memory, slightly less accurate
- **DiskANN**: SSD-based, for billion-scale datasets

**Q13: What is HNSW and how does it work?**
**A**: HNSW builds a multi-layer graph structure. The top layer is sparse (long-range connections), bottom layer is dense (nearby neighbors). Search starts at the top layer and descends, quickly narrowing the search space. Key parameters: M (connections per node), efConstruction (build quality), ef (search depth).

**Q14: What's the difference between cosine similarity, Euclidean distance, and dot product?**
**A**:
- **Cosine similarity**: measures angle between vectors (1 = same direction, -1 = opposite). Best for text embeddings.
- **Euclidean (L2)**: measures straight-line distance. Best for image embeddings.
- **Dot product**: measures magnitude × cos(angle). Best for normalized vectors (same as cosine).
- For normalized vectors: cosine = dot product, L2 ∝ 1 - cosine.

**Q15: How do you choose a distance metric?**
**A**: 
- Text embeddings → **cosine** (standard, captures semantic direction)
- Image embeddings → **L2** (Euclidean, captures visual differences)
- Normalized embeddings → **dot product** (equivalent to cosine, faster)
- When in doubt → **cosine**

**Q16: What is the curse of dimensionality in vector search?**
**A**: As dimensions increase, distances between points become more uniform (all points look equally far apart). This makes nearest neighbor search harder. Common solutions: use dimensionality reduction (PCA), product quantization, or specialized ANN algorithms. Most text embeddings (384-1536 dims) don't suffer severely from this.

**Q17: How do you scale a vector database to millions of vectors?**
**A**: 
1. Use proper ANN indexing (HNSW with tuned parameters)
2. Partition by metadata (sharding)
3. Use quantization (reduce precision: 32-bit → 8-bit)
4. Horizontal scaling (multiple nodes with replication)
5. Consider disk-based ANN (DiskANN) for billion-scale

**Q18: What is ChromaDB and what are its limitations?**
**A**: ChromaDB is an open-source, Python-native vector database. Pros: easy setup, no separate server, good for prototyping. Cons: single-node, limited to medium scale (<1M vectors), no built-in replication/sharding. Good for development, consider Qdrant/Weaviate for production.

**Q19: Explain metadata filtering in vector search.**
**A**: Two approaches:
- **Pre-filter**: Apply metadata filter first, then search within filtered set. Better for selective queries.
- **Post-filter**: Search all vectors, then filter results. Better for high recall.
- Modern vector DBs support hybrid: pre-filtering integrated into ANN search.

**Q20: How do you handle multi-tenancy in vector databases?**
**A**: 
- Separate collections/namespaces per tenant
- Metadata filtering by tenant_id (pre-filter)
- Partitioned indexes
- Choose a vector DB with native multi-tenancy support (Qdrant, Pinecone)
