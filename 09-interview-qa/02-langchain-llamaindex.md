# Interview Q&A — LangChain & LlamaIndex

## LangChain

**Q1: What is LangChain and what problem does it solve?**
**A**: LangChain is a framework for building LLM-powered applications. It solves the problem of **glue code**: instead of manually writing API calls, prompt templates, and state management, LangChain provides standardized abstractions (LLM wrappers, prompt templates, chains, memory, agents) that compose together. It's the "React/Angular" for LLM apps.

**Q2: What are the core components of LangChain?**
**A**:
1. **Models**: LLM wrappers (Ollama, OpenAI, HuggingFace)
2. **Prompts**: PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
3. **Chains**: LLMChain, SequentialChain, RouterChain
4. **Memory**: BufferMemory, WindowMemory, SummaryMemory
5. **Agents**: Agent + Tool combinations (ReAct, OpenAI Tools)
6. **Retrievers**: Vector store retrieval, document loaders
7. **Callbacks**: Logging, monitoring, streaming

**Q3: What is LCEL (LangChain Expression Language)?**
**A**: LCEL is the modern way to compose LangChain components using the `|` operator (Unix pipe style). Example:
```python
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "RAG"})
```
LCEL provides streaming, async, parallel execution, and better error handling than the older Chain APIs.

**Q4: How does memory work in LangChain?**
**A**: Memory stores conversation history and injects it into prompts. Types:
- **ConversationBufferMemory**: stores full history (expensive)
- **ConversationBufferWindowMemory**: keeps only last K turns
- **ConversationSummaryMemory**: LLM-summarized history
- **VectorStoreMemory**: RAG-like retrieval from past conversation

All work by adding formatted history to the prompt as context.

**Q5: What is an agent in LangChain?**
**A**: An agent is an LLM that can decide which tools to call and in what order. It follows the **ReAct** pattern: Reasoning (thinking about what to do) + Acting (calling tools) + Observing (processing results). The agent iterates until it can provide a final answer or hits the iteration limit.

**Q6: How do you create a custom tool in LangChain?**
**A**: A tool is a function with a name and description. The LLM reads the description to decide when to use it.
```python
from langchain.agents import Tool

def my_function(input: str) -> str:
    return f"Processed: {input}"

tool = Tool(
    name="MyTool",
    func=my_function,
    description="Useful when you need to process X. Input should be..."
)
```
The description is critical — it's the LLM's guide for when to call this tool.

**Q7: What's the difference between a chain and an agent?**
**A**: A **chain** is a fixed sequence of steps (A → B → C). An **agent** dynamically decides what to do next based on the current state. Chains are predictable and fast; agents are flexible but slower and less predictable.

**Q8: How do you stream responses in LangChain?**
**A**: LCEL supports streaming natively:
```python
for chunk in chain.stream({"topic": "RAG"}):
    print(chunk, end="")
```
Or use `.astream()` for async. Callbacks also support per-token streaming.

## LlamaIndex

**Q9: What is LlamaIndex and how is it different from LangChain?**
**A**: LlamaIndex specializes in **data indexing and retrieval** for LLMs. LangChain is a general-purpose LLM framework that happens to include retrieval. LlamaIndex has deeper support for:
- Multiple data sources (PDFs, databases, APIs)
- Advanced indexing strategies (tree, keyword, vector, hybrid)
- Query decomposition and routing
- Structured data extraction

**Q10: What are the core concepts in LlamaIndex?**
**A**:
- **Documents**: Raw data sources (files, web pages)
- **Nodes**: Chunks of text with metadata (parsed from documents)
- **Indices**: Organized structures for retrieval (VectorStoreIndex, SummaryIndex, TreeIndex)
- **Query Engines**: Retrieve + synthesize answers from indices
- **Retrievers**: Extract relevant nodes from indices
- **Response Synthesizers**: Combine retrieved nodes into LLM answers

**Q11: Explain the indexing process in LlamaIndex.**
**A**: 1. Load documents (SimpleDirectoryReader) → 2. Parse into nodes (SentenceSplitter, with chunk size and overlap) → 3. Embed each node (embedding model) → 4. Store embeddings in vector index → 5. Optional: persist index to disk.

**Q12: What chunking strategies does LlamaIndex support?**
**A**: 
- **SentenceSplitter**: splits at sentence boundaries (default, recommended)
- **TokenTextSplitter**: splits at token count boundaries
- **RecursiveCharacterTextSplitter**: tries paragraph, then sentence, then character
- **SemanticSplitterNodeParser**: splits based on embedding similarity (advanced)

Chunk size and overlap are critical parameters. Smaller chunks = more precise retrieval but less context. 20% overlap prevents splitting relevant text across chunks.

**Q13: What is a SubQuestionQueryEngine?**
**A**: It breaks a complex question into sub-questions, routes each to the appropriate index/tool, and synthesizes the answers. For example, "What is Python and how is it used in ML?" becomes "What is Python?" → Python index + "How is Python used in ML?" → ML index.

**Q14: How does LlamaIndex handle structured data?**
**A**: Through **structured indices** that use LLMs to extract structured information from documents. The `StructuredExtractor` and `PropertyGraphIndex` can extract entities, relationships, and properties from unstructured text and store them in a structured format.

**Q15: What's the difference between a query engine and a retriever?**
**A**: A **retriever** finds relevant nodes from an index and returns them. A **query engine** wraps a retriever + response synthesizer: it retrieves nodes, then uses an LLM to generate an answer from those nodes. Query engines are the "end-to-end" interface for asking questions.
