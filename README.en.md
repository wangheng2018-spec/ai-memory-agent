# 🧠 ai-memory-agent — Agent Memory System

> Stage 3 · Lesson 1: Teaching AI customer service to "remember customers"
> The previous lesson (ai-knowledge-shop) taught AI to answer questions using a **knowledge base**; this lesson teaches AI to remember **who each customer is**.

## ✨ Impact Comparison

| Scenario | Standard Support (Previous Lesson) | Memory-Enhanced Support (This Lesson) |
|----------|-----------------------------------|---------------------------------------|
| Returning customer asks "Can you do better on that one from last time?" | "Which one? Please describe it again" | "Mr. Zhang, you're asking about the 16 Pro Max 512G — 95% battery health, reference price ¥6100. Let me see what I can do for you." |
| Customer profile | None | Preferred models / budget / concerns / intent level — gets more accurate with each interaction |
| Cross-customer operations | None | "Find all high-intent customers wanting 512G" — one semantic query |

## 🏗️ Three-Layer Memory Architecture

```
┌─────────────────────────────────────────────────┐
│  Receive Customer Message                        │
│   ├─ ① Recall (Episodic Memory)  ChromaDB reads customer profile + history │
│   ├─ ② Retrieve (Semantic Memory)  RAG knowledge base (reused from previous lesson) │
│   ├─ ③ Generate (Working Memory)  DeepSeek + conversation context │
│   └─ ④ Consolidate           Conversation ends → distill → update memory │
└─────────────────────────────────────────────────┘
```

- **Working Memory**: Short-term context within the current session, stored as a Python dict, keeping only the most recent N turns
- **Episodic Memory**: Long-term customer memory stored in ChromaDB, accessed precisely by customer ID (`cust_{id}` as one document). Content = customer profile + historical consultation summaries
- **Semantic Memory**: Store knowledge base with RAG retrieval (keyword candidates + bge-reranker re-ranking)

## 🚀 Getting Started

```bash
# 1. Set API Keys (DeepSeek for chat; SiliconFlow for embedding/rerank)
# Windows PowerShell:
$env:DEEPSEEK_API_KEY = "sk-..."
$env:SILICONFLOW_API_KEY = "sk-..."

# 2. Test storage layer (optional)
python memory_store.py

# 3. Full demo: the same customer returns days later, and the agent remembers them
python agent.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | Chat LLM (official DeepSeek; automatically falls back to SiliconFlow DeepSeek if invalid) |
| `SILICONFLOW_API_KEY` | Embedding (bge-m3) + re-ranking (bge-reranker-v2-m3) |
| `MEMORY_EMBED_BACKEND` | `siliconflow` (default, better quality) or `local` (Chroma's built-in all-MiniLM-L6-v2, free & offline, weaker for Chinese) |

## 📁 Project Structure

```
ai-memory-agent/
├── memory_store.py   # Storage layer: WorkingMemory / EpisodicMemory / vector storage
├── agent.py          # Agent layer: recall → retrieve → generate → consolidate full pipeline + demo
└── chroma_db/        # ChromaDB persistence (customer memory store)
```

## 🛠️ Key Technical Points

1. **Memory = Document**: Each customer is one Chroma document (profile + history); `upsert` natively supports "memory growth"
2. **Precise Recall vs. Semantic Recall**: Use ID-based `get()` for exact retrieval; use `query()` for cross-customer semantic search
3. **Memory Grows**: After each conversation, `remember()` distills new information into the profile and history
4. **Profile Distillation**: Heuristic regex extracts preferred models / budget / concerns / intent level from conversations (production can swap in LLM-based distillation)
5. **Graceful Degradation**: If SiliconFlow runs out of credits (402), automatically falls back to local embedding — core functionality never breaks

## 📌 Pitfalls & Lessons Learned

- **"Bearer " gets redacted by the security layer**: Code containing Authorization headers written via the write tool gets replaced with `***` — always verify with `Select-String` after writing
- **SiliconFlow free tier limits**: bge-m3 has a free quota that eventually runs out; after that you'll get 402 errors — top up or fall back to `local`
- **DeepSeek keys can expire**: This project implements dual-channel automatic fallback (official → SiliconFlow DeepSeek)
- **Windows console encoding issues**: Run `$env:PYTHONIOENCODING="utf-8"` before execution; `python -X utf8` is even more reliable

## 🔗 Related Projects

- Previous lesson: ai-knowledge-shop (Advanced RAG engineering: BM25 / vector store / hybrid retrieval / neural re-ranking)
- Next lesson preview: Agent planning capabilities (ReAct / Plan-and-Execute)