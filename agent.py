# -*- coding: utf-8 -*-
"""
Agent 记忆系统 · 智能体层
==========================
阶段3 · 第1课：让 AI 客服"记得客户"

完整链路（这就是一个带记忆的 Agent）：
  收到消息
   ├─ ① 回忆：从 ChromaDB 读这位客户的历史记忆（画像+历史咨询）
   ├─ ② 检索：RAG 从店铺知识库找相关资料（复用上一课混合检索）
   ├─ ③ 生成：DeepSeek 结合【客户记忆】+【知识库】+【当前对话】回答
   └─ ④ 写入：对话结束后，自动提炼新信息，更新客户记忆（记忆会成长）

对比上一课 shop-ai-cs：
  之前：每个客户都是"陌生人"，同样的价格问 10 遍答 10 遍
  现在：老客户一开口，客服就知道 TA 是谁、上次聊到哪、在意什么

运行: python agent.py （需要 DEEPSEEK_API_KEY + SILICONFLOW_API_KEY）
"""

import os
import re
import sys
import time
import requests

from memory_store import WorkingMemory, EpisodicMemory, build_memory_context

# ============ 配置 ============
# 通道1：DeepSeek 官方
DS_API_URL = "https://api.deepseek.com/chat/completions"
DS_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DS_MODEL = "deepseek-chat"

# 通道2：SiliconFlow 上的 DeepSeek（key 失效时自动切换）
SF_API_URL = "https://api.siliconflow.cn/v1"
SF_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SF_DS_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 店铺知识库（跟上一课一致）
KNOWLEDGE_BASE = [
    {"id": "doc1", "category": "售后", "title": "保修政策",
     "content": "本店所有手机提供7天无理由退换，15天内质量问题免费换机，1年店铺保修。人为损坏（进水、摔碎、私自拆机）不在保修范围内。"},
    {"id": "doc2", "category": "交易", "title": "交易方式",
     "content": "支持闲鱼平台担保交易，同城支持华强北面交，外地顺丰包邮发货。发货前会录制验机视频，全程可追溯。"},
    {"id": "doc3", "category": "售后", "title": "退货流程",
     "content": "退货流程：先联系客服说明原因，拍照留证，7天内寄回。确认无人为损坏后，24小时内退款到原支付渠道。"},
    {"id": "doc4", "category": "产品", "title": "iPhone 16 Pro Max 256G",
     "content": "iPhone 16 Pro Max 256G 二手95新，电池效率95%，参考成交价5400元。支持5G双卡，A18 Pro芯片，钛金属边框。"},
    {"id": "doc5", "category": "产品", "title": "iPhone 16 Pro Max 512G",
     "content": "iPhone 16 Pro Max 512G 二手95新，电池效率95%，参考成交价6100元。超大存储适合拍视频和存大量照片。"},
    {"id": "doc6", "category": "产品", "title": "iPhone 15 Pro Max 256G",
     "content": "iPhone 15 Pro Max 256G 二手95新，电池效率95%，参考成交价5300元。A17 Pro芯片，USB-C接口，钛金属设计。"},
    {"id": "doc7", "category": "产品", "title": "iPhone 13 128G",
     "content": "iPhone 13 128G 二手9成新，电池效率88%，参考成交价2600元。入门款性价比之选，日常使用完全够用。"},
    {"id": "doc8", "category": "售后", "title": "换机政策",
     "content": "15天内出现非人为质量问题，支持免费换同款同色新机。换机需保留原包装和配件，到店或邮寄均可。"},
    {"id": "doc9", "category": "交易", "title": "付款安全",
     "content": "请务必通过闲鱼平台下单付款，不要私下转账。私下转账无法保障权益，遇到要求微信直接转账的一律是骗子。"},
    {"id": "doc10", "category": "产品", "title": "电池说明",
     "content": "二手手机电池效率在88%-95%之间。电池效率低不等于手机有问题，只是续航稍短，可另付费更换原装电池（150元）。"},
]


# ============ 1. RAG 检索（知识库） ============
def retrieve(query, top_k=3):
    """轻量混合检索：关键词(jieba) 候选 + bge-reranker 精排"""
    import jieba
    words = set(jieba.lcut(query))
    scored = []
    for item in KNOWLEDGE_BASE:
        hit = 0
        for kw in item["content"].split(" "):
            pass
        # 简单关键词打分：查内容里的关键实体
        kws = set(re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z0-9]+", item["content"]))
        inter = words & kws
        if inter:
            hit = len(inter)
        if hit > 0:
            scored.append((hit, item))
    scored.sort(key=lambda x: -x[0])
    candidates = [item for _, item in scored[:6]] or KNOWLEDGE_BASE[:2]
    # 精排
    try:
        r = requests.post(
            f"{SF_API_URL}/rerank",
            headers={"Authorization": "Bearer " + SF_API_KEY},
            json={"model": RERANK_MODEL, "query": query,
                  "documents": [c["content"] for c in candidates], "top_n": top_k},
            timeout=30,
        )
        r.raise_for_status()
        out = []
        for item in r.json()["results"]:
            c = candidates[item["index"]]
            c["rerank_score"] = round(item["relevance_score"], 4)
            out.append(c)
        return out
    except Exception:
        return candidates[:top_k]


# ============ 2. 调 LLM（双通道：DeepSeek 官方 → SiliconFlow 兜底） ============
def call_llm(messages, temperature=0.5, max_tokens=500, retries=2):
    """优先 DeepSeek 官方；key 失效/报错时自动切 SiliconFlow 的 DeepSeek 模型"""
    channels = []
    if DS_API_KEY:
        channels.append((DS_API_URL, DS_API_KEY, DS_MODEL, "DeepSeek官方"))
    if SF_API_KEY:
        channels.append((SF_API_URL + "/chat/completions", SF_API_KEY, SF_DS_MODEL, "SiliconFlow"))
    if not channels:
        raise RuntimeError("未配置任何 LLM API Key")

    last_err = None
    for url, key, model, name in channels:
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        for attempt in range(retries):
            try:
                r = requests.post(url, json=payload, headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                }, timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                # 401/403 = key 失效，直接切下一个通道
                if e.response is not None and e.response.status_code in (401, 403):
                    last_err = e
                    print(f"  ⚠️ {name} key 失效({e.response.status_code})，切换通道...")
                    break
                if attempt == retries - 1:
                    last_err = e
                else:
                    time.sleep(1)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == retries - 1:
                    last_err = e
                else:
                    time.sleep(2 * (attempt + 1))
    raise last_err


# ============ 3. 记忆 Agent ============
SYSTEM_PROMPT = """你是『老王手机』店铺的 AI 客服，老板卖 iPhone 为主。
你的特点：**记得每一位客户**。

【如何用记忆】
1. 如果【这位客户的历史记忆】里有 TA 的画像或历史咨询：
   - 主动体现"记得TA"：比如"张哥，上次您问的 512G 那台考虑得怎么样？"
   - 结合历史信息回答，别让客户重复说
2. 如果是新客户：正常热情接待，并在聊的过程中自然了解需求
3. 涉及价格：用【店铺知识库】里的参考成交价，并引导客户提供 型号+存储+电池效率
4. 涉及交易/售后/付款：用知识库的资料回答，不要编造
5. 资料没有的，诚实说"这个我帮您问下老板"
6. 语气口语化、真诚、简短（一般 2~4 句），emoji 不超过 3 个

【这位客户的历史记忆】
{memory}

【店铺知识库资料】
{context}"""


class MemoryAgent:
    def __init__(self, customer_id, name=None):
        self.customer_id = customer_id
        self.name = name
        self.episodic = EpisodicMemory(customer_id, name)
        self.working = WorkingMemory(max_turns=8)
        # 会话内累积的"新记忆素材"
        self._new_history = []

    def _load_old_memory(self):
        mem = self.episodic.load()
        self._old_history = []
        if mem:
            # 从记忆文本里提取历史咨询行（只取客户问的，避免重复叠加客服答）
            for line in mem["text"].split("\n"):
                line = line.strip()
                if line.startswith("- ") and "客服答" not in line:
                    self._old_history.append(line[2:])
        return mem

    # ---- 核心：处理一条消息 ----
    def respond(self, user_msg):
        # ① 工作记忆加入本次提问
        self.working.add("user", user_msg)

        # ② 回忆长期记忆
        mem = self._load_old_memory()

        # ③ RAG 检索知识库
        hits = retrieve(user_msg, top_k=3)
        context = "\n".join(
            f"[{h['category']}] {h['title']}: {h['content']}" for h in hits
        )

        # ④ 拼记忆上下文
        memory_text, _ = build_memory_context(
            self.customer_id, self.name, self.working
        )

        # ⑤ 生成回答
        sys_prompt = SYSTEM_PROMPT.format(memory=memory_text, context=context)
        messages = [
            {"role": "system", "content": sys_prompt},
        ] + [
            {"role": "user" if t["role"] == "user" else "assistant", "content": t["text"]}
            for t in self.working.recent(6)
        ]
        reply = call_llm(messages)
        self.working.add("assistant", reply)

        # ⑥ 记录新记忆素材（简短摘要）
        self._new_history.append(f"{time.strftime('%m/%d')} 客户问：{user_msg[:40]}")
        if reply:
            self._new_history.append(f"  客服答：{reply[:40]}")

        return reply, hits

    # ---- 记忆写入：对话结束后调用 ----
    def remember(self, summary=""):
        """把本次对话沉淀进长期记忆（画像 + 历史）"""
        # 旧历史 + 新历史
        all_history = self._old_history + self._new_history
        # 画像：从对话里提炼（简单启发式）
        profile = self._extract_profile(all_history)
        self.episodic.save(profile=profile, history=all_history)

    @staticmethod
    def _extract_profile(history):
        """从历史咨询中提炼客户画像（规则版；生产可用 LLM 提炼）"""
        profile = {}
        text = "\n".join(history)
        # 偏好机型（兼容无 iPhone 前缀的写法："16 Pro Max 512G"，排除日期/时间）
        # 先去掉日期时间（如 08/28、8/20）
        clean = re.sub(r"\d{1,2}/\d{1,2}", "", text)
        m = re.search(r"(?:iPhone\s*)?(\d{2,3}\s*(?:Pro\s*Max|Pro|Plus|mini)?\s*\d*G?)", clean, re.I)
        if m:
            model = m.group(1).strip()
            if re.match(r"^\d{2,3}", model):  # 确认是机型（2-3位数字开头）
                profile["偏好机型"] = ("iPhone " + model) if not re.match(r"^iPhone", model, re.I) else model
        # 预算
        m = re.search(r"(?:预算|价位|心理价|多少[钱价])(?:在|是|)??[^，。]{0,6}?(\d{3,5})", text)
        if m:
            profile["预算区间"] = m.group(1) + "左右"
        # 关注点
        cares = []
        for kw in ["电池", "保修", "成色", "发票", "便宜", "砍价", "分期"]:
            if kw in text:
                cares.append(kw)
        if cares:
            profile["关注点"] = "、".join(cares[:4])
        # 意向度（排除"成交价"这种误匹配）
        if re.search(r"(已成交|成交了|已下单|已购买|已付款)", text):
            profile["意向度"] = "已成交客户"
        elif "考虑" in text or "商量" in text or "再便宜" in text or "优惠" in text:
            profile["意向度"] = "高意向（考虑中）"
        return profile


# ============ 4. 命令行演示 ============
def demo():
    if not DS_API_KEY or not SF_API_KEY:
        print("需要设置 DEEPSEEK_API_KEY 和 SILICONFLOW_API_KEY")
        sys.exit(1)

    print("=" * 66)
    print("Agent 记忆系统 · 演示：同一个客户，隔几天再来，客服记得TA")
    print("=" * 66)

    cid = "wx_demo_001"
    agent = MemoryAgent(cid, name="张哥")

    print("\n───── 第 1 天：首次咨询 ─────")
    q1 = "老板，16 Pro Max 512G 多少钱？电池怎么样？"
    print(f"👤 客户: {q1}")
    reply, hits = agent.respond(q1)
    print(f"🤖 客服: {reply}")

    print("\n💾 对话结束，客服把今天聊的写进长期记忆...")
    agent.remember()

    print("\n───── 第 3 天：客户又来了（新会话，但记忆还在）─────")
    # 模拟 3 天后的新会话：重新建 agent（等于重启服务/新会话）
    agent2 = MemoryAgent(cid, name="张哥")
    q2 = "在吗？上次那个 512G 的还能便宜点不？"
    print(f"👤 客户: {q2}")
    reply2, _ = agent2.respond(q2)
    print(f"🤖 客服: {reply2}")

    print("\n💾 再次沉淀记忆...")
    agent2.remember()

    print("\n───── 最终客户记忆（查看成长效果）─────")
    mem = agent2.episodic.load()
    print(mem["text"] if mem else "  (空)")

    print("\n───── 跨客户语义回忆：找高意向客户 ─────")
    hits = EpisodicMemory.recall("预算6000想买大内存的高意向客户", top_k=3)
    for h in hits:
        name = h["meta"].get("name", "?")
        print(f"  {h['score']:.3f} [{name}] {h['text'][:60].replace(chr(10),' ')}...")


if __name__ == "__main__":
    demo()
