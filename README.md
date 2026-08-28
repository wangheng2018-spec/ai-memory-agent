# 🧠 ai-memory-agent — Agent 记忆系统

> 阶段3 · 第1课：让 AI 客服"记得客户"
> 上一课（ai-knowledge-shop）教会 AI 用**知识库**回答问题；这一课让 AI 记住**每一个客户是谁**。

## ✨ 效果对比

| 场景 | 普通客服（上一课） | 记忆客服（本课） |
|------|------------------|-----------------|
| 老客户问"上次那台还能便宜吗" | "哪台？请重新描述" | "张哥，您问的 16 Pro Max 512G 那台，95新电池95%，参考价6100，我帮您跟老板磨磨" |
| 客户画像 | 无 | 偏好机型 / 预算 / 关注点 / 意向度，越聊越准 |
| 跨客户运营 | 无 | "找出所有想要512G的高意向客户" 一条语义检索 |

## 🏗️ 三层记忆架构

```
┌─────────────────────────────────────────────────┐
│  收到客户消息                                      │
│   ├─ ① 回忆(情景记忆)  ChromaDB 读客户画像+历史     │
│   ├─ ② 检索(语义记忆)  RAG 知识库(复用上一课)       │
│   ├─ ③ 生成(工作记忆)  DeepSeek + 对话上下文        │
│   └─ ④ 沉淀           对话结束 → 提炼 → 更新记忆    │
└─────────────────────────────────────────────────┘
```

- **工作记忆 (WorkingMemory)**：当前会话内的短期上下文，Python dict，只保留最近 N 轮
- **情景记忆 (EpisodicMemory)**：长期客户记忆，存 ChromaDB，按客户ID精确存取（`cust_{id}` 一条文档），内容 = 客户画像 + 历史咨询摘要
- **语义记忆**：店铺知识库，RAG 检索（关键词候选 + bge-reranker 精排）

## 🚀 运行

```bash
# 1. 设置 Key（DeepSeek 用于对话；SiliconFlow 用于 embedding/rerank）
# Windows PowerShell:
$env:DEEPSEEK_API_KEY = "sk-..."
$env:SILICONFLOW_API_KEY = "sk-..."

# 2. 存储层自测（可选）
python memory_store.py

# 3. 完整演示：同一个客户隔几天再来，客服记得TA
python agent.py
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | 对话 LLM（DeepSeek 官方，失效时自动 fallback SiliconFlow DeepSeek） |
| `SILICONFLOW_API_KEY` | embedding (bge-m3) + 重排 (bge-reranker-v2-m3) |
| `MEMORY_EMBED_BACKEND` | `siliconflow`（默认，效果好）或 `local`（Chroma 内置 all-MiniLM-L6-v2，免费离线，中文一般） |

## 📁 文件

```
ai-memory-agent/
├── memory_store.py   # 存储层：WorkingMemory / EpisodicMemory / 向量存取
├── agent.py          # 智能体层：回忆→检索→生成→沉淀 完整链路 + 演示
└── chroma_db/        # ChromaDB 持久化（客户记忆库）
```

## 🛠️ 关键技术点

1. **记忆 = 文档**：每个客户一条 Chroma 文档（画像+历史），`upsert` 天然支持"记忆成长"
2. **精确回忆 vs 语义回忆**：按 ID `get()` 精确读；跨客户用 `query()` 语义找
3. **记忆会成长**：每次对话结束，`remember()` 把新信息提炼进画像和历史
4. **画像提炼**：启发式正则从对话里抽 偏好机型/预算/关注点/意向度（生产可换 LLM 提炼）
5. **降级容错**：SiliconFlow 余额不足(402)自动降级本地 embedding，核心功能不中断

## 📌 踩坑记录

- **"Bearer " 被安全层替换**：write 工具写含 Authorization 的代码会被替换成 `***`，写完必须 `Select-String` 检查
- **SiliconFlow 免费额度**：bge-m3 有免费额度但会耗尽，耗尽后 402，需充值或降级 local
- **DeepSeek key 会失效**：本项目做了双通道自动 fallback（官方 → SiliconFlow DeepSeek）
- **Windows 控制台乱码**：运行前 `$env:PYTHONIOENCODING="utf-8"`；`python -X utf8` 更稳

## 🔗 关联作品

- 上一课：ai-knowledge-shop（RAG 深度工程：BM25/向量库/混合检索/神经重排）
- 下一课预告：Agent 规划能力（ReAct / Plan-and-Execute）
