# -*- coding: utf-8 -*-
"""
Agent 记忆系统 · 存储层
========================
阶段3 · 第1课：让 AI 客服"记得客户"

三类记忆：
  1. 工作记忆 (working)  —— 当前会话内的短期上下文（Python dict，会话结束就丢）
  2. 情景记忆 (episodic) —— 长期客户记忆，存 ChromaDB，按客户ID存取
     · 客户画像（昵称、偏好机型、预算区间、是否老客户）
     · 历史咨询记录（每次问过什么、报过什么价、成交没有）
  3. 语义记忆 (semantic) —— 原有店铺知识库（RAG 检索用），复用上一课成果

为什么用向量库存"记忆"而不是普通数据库？
  → 记忆需要"回忆"：客户说"上次那个大内存的机器"，系统要能语义匹配到
    "512G 的 iPhone 16 Pro Max"——关键词匹配做不到，向量相似度可以。

运行: 本文件是库，配合 agent.py / app.py 使用
"""

import os
import time
import requests
import chromadb

# ============ 配置 ============
SF_API_URL = "https://api.siliconflow.cn/v1"
SF_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
EMBED_MODEL = "BAAI/bge-m3"  # 中文语义 embedding（1024维）

# embedding 通道：
#   "siliconflow" → 调 bge-m3 API（效果最好，需 SiliconFlow 余额）
#   "local"       → Chroma 内置 all-MiniLM-L6-v2（免费离线，首次下载模型，中文效果一般）
# 余额不足时自动降级 local（下面的 SiliconFlowEmbedding 会捕获 402 异常）
EMBED_BACKEND = os.environ.get("MEMORY_EMBED_BACKEND", "siliconflow")  # siliconflow | local

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


# ============ 中文 embedding 函数（复用上一课） ============
class SiliconFlowEmbedding:
    """把硅基流动 bge-m3 变成 Chroma 能用的 embedding 函数。
    余额不足(402)时自动降级为本地 all-MiniLM-L6-v2。
    """
    def __init__(self, api_key, model=EMBED_MODEL):
        self.api_key = api_key
        self.model = model
        self._cache = {}
        self._local = None  # 懒加载本地模型

    def _get_local(self):
        if self._local is None:
            # 用 Chroma 自带的默认 embedding（all-MiniLM-L6-v2，离线可用）
            from chromadb.utils import embedding_functions
            self._local = embedding_functions.DefaultEmbeddingFunction()
        return self._local

    def __call__(self, input):
        if EMBED_BACKEND == "local":
            return self._get_local()(input)
        texts = input if isinstance(input, list) else [input]
        vectors = []
        for t in texts:
            if t in self._cache:
                vectors.append(self._cache[t])
                continue
            try:
                r = requests.post(
                    f"{SF_API_URL}/embeddings",
                    headers={"Authorization": "Bearer " + self.api_key},
                    json={"model": self.model, "input": [t]},
                    timeout=30,
                )
                r.raise_for_status()
                vec = r.json()["data"][0]["embedding"]
                self._cache[t] = vec
                vectors.append(vec)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (401, 402, 403):
                    # 余额不足/key失效 → 降级本地 embedding
                    print(f"  ⚠️ SiliconFlow embedding 不可用({e.response.status_code})，降级本地 all-MiniLM-L6-v2")
                    return self._get_local()(input)
                raise
        return vectors


# ============ Chroma 客户端（单例） ============
_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def get_collection():
    """情景记忆集合：每个客户一条记忆文档"""
    global _collection
    if _collection is None:
        client = get_client()
        try:
            _collection = client.get_collection(
                name="customer_memories",
                embedding_function=SiliconFlowEmbedding(SF_API_KEY),
            )
        except Exception:
            _collection = client.create_collection(
                name="customer_memories",
                embedding_function=SiliconFlowEmbedding(SF_API_KEY),
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


# ============ 1. 工作记忆（短期） ============
class WorkingMemory:
    """当前会话的短期记忆：就是个带上下文的 dict"""
    def __init__(self, max_turns=8):
        self.turns = []        # [{"role": "user"/"assistant", "text": ...}]
        self.max_turns = max_turns
        self.attrs = {}        # 会话内临时属性（比如正在谈的机型）

    def add(self, role, text):
        self.turns.append({"role": role, "text": text})
        # 只保留最近 N 轮，防止上下文无限膨胀
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def recent(self, n=6):
        """取最近 n 条消息"""
        return self.turns[-n:]

    def summary_text(self, n=6):
        """拼成给 LLM 的对话历史文本"""
        lines = []
        for t in self.recent(n):
            who = "客户" if t["role"] == "user" else "客服"
            lines.append(f"{who}: {t['text']}")
        return "\n".join(lines)

    def clear(self):
        self.turns = []
        self.attrs = {}


# ============ 2. 情景记忆（长期，存 ChromaDB） ============
class EpisodicMemory:
    """
    每个客户一条记忆记录，存在 ChromaDB：
      document = 客户画像 + 历史咨询摘要（自然语言，可被语义检索）
      metadata = {customer_id, name, updated_at, ...}

    为什么整个存一条而不是一次咨询存一条？
      → 更简单：读出来就是完整的"这个人是谁"。向量检索还能做
        "找所有想要 512G 的客户"这类跨客户语义查询。
    """

    def __init__(self, customer_id, name=None):
        self.customer_id = customer_id
        self.name = name
        self.col = get_collection()

    # ---- 构造记忆文本 ----
    def _memory_text(self, profile, history):
        parts = []
        # 画像
        parts.append("【客户画像】")
        if self.name:
            parts.append(f"昵称: {self.name}")
        for k, v in profile.items():
            parts.append(f"{k}: {v}")
        # 历史咨询
        if history:
            parts.append("【历史咨询】")
            for h in history[-10:]:  # 最多带最近10条
                parts.append(f"- {h}")
        return "\n".join(parts)

    # ---- 保存（写记忆） ----
    def save(self, profile=None, history=None):
        """写入/更新该客户在 ChromaDB 的记忆"""
        profile = profile or {}
        history = history or []
        text = self._memory_text(profile, history)
        doc_id = f"cust_{self.customer_id}"
        self.col.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[{
                "customer_id": self.customer_id,
                "name": self.name or "",
                "updated_at": int(time.time()),
                "turns": len(history),
            }],
        )

    # ---- 读取（回忆） ----
    def load(self):
        """按客户ID精确读回记忆"""
        try:
            res = self.col.get(ids=[f"cust_{self.customer_id}"])
        except Exception:
            return None
        if res and res["ids"]:
            return {
                "id": res["ids"][0],
                "text": res["documents"][0],
                "meta": res["metadatas"][0],
            }
        return None

    # ---- 语义回忆（跨客户模糊搜索） ----
    @staticmethod
    def recall(query, top_k=3):
        """语义检索所有客户记忆：找"想要512G的客户""上次问过价格的"
        返回 [{id, text, meta, score}]
        """
        col = get_collection()
        if col.count() == 0:
            return []
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "meta": results["metadatas"][0][i],
                "score": round(1 - results["distances"][0][i], 4),
            })
        return hits

    # ---- 删除 ----
    def forget(self):
        try:
            self.col.delete(ids=[f"cust_{self.customer_id}"])
            return True
        except Exception:
            return False


# ============ 3. 记忆汇总：给 LLM 看 ============
def build_memory_context(customer_id, name=None, working=None):
    """
    把三类记忆合成一段给 LLM 的"记忆上下文"：
      【长期记忆】客户画像 + 历史咨询（情景记忆，ChromaDB）
      【本次对话】最近几轮（工作记忆）
    """
    parts = []

    # 情景记忆（长期）
    epi = EpisodicMemory(customer_id, name)
    mem = epi.load()
    if mem:
        parts.append("【这位客户的历史记忆】（记得TA！）")
        parts.append(mem["text"])
    else:
        parts.append("【这位客户的历史记忆】（首次咨询，无历史记录）")

    # 工作记忆（短期）
    if working and working.turns:
        parts.append("\n【本次对话进行中】")
        parts.append(working.summary_text(6))

    return "\n".join(parts), mem


if __name__ == "__main__":
    print("=" * 60)
    print("记忆系统 · 存储层自测")
    print("=" * 60)

    # 模拟两个客户
    print("\n[1] 写入客户 A 的记忆...")
    a = EpisodicMemory("wx_1001", name="张哥")
    a.save(
        profile={"偏好机型": "iPhone 16 Pro Max", "预算": "6000左右", "关注点": "电池效率、保修"},
        history=[
            "8/20 咨询 16 Pro Max 512G，报价6100，说考虑一下",
            "8/25 又问 16 Pro Max 256G 价格，报价5400，问能否便宜",
            "还没成交，属于高意向客户",
        ],
    )
    print("  已保存")

    print("\n[2] 写入客户 B 的记忆...")
    b = EpisodicMemory("wx_1002", name="李姐")
    b.save(
        profile={"偏好机型": "iPhone 13/14 系列", "预算": "3000以内", "关注点": "性价比"},
        history=[
            "8/22 咨询 iPhone 13 128G，报价2600，说回去和老公商量",
            "8/27 回来问 13 还能再便宜吗，告知可小刀，未成交",
        ],
    )
    print("  已保存")

    print("\n[3] 按ID读回客户 A 的记忆（精确回忆）...")
    mem_a = a.load()
    print(mem_a["text"] if mem_a else "  (空)")

    print("\n[4] 语义回忆测试：'想要512G大内存的客户'...")
    hits = EpisodicMemory.recall("想要512G大内存的客户", top_k=2)
    for h in hits:
        print(f"  {h['score']:.3f} [{h['meta']['name']}] {h['text'][:50]}...")

    print("\n[5] 删除客户 B...")
    b.forget()
    print("  已删除，剩余客户数:", get_collection().count())
