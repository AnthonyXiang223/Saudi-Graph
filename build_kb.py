"""
MAZU Knowledge Base Builder — 离线构建向量检索库

输入:
  docs/*.md          — 6 个技术文档（中文）
  schema/operators.json — 91 个气象指标定义
  schema/rules.json     — 4 条灾害检测规则
  agent.py              — 系统 Prompt 中的沙特气候研判知识

输出:
  kb/  — ChromaDB 向量库 (~300 个文档块, embedding 维度 384)

用法:
  python build_kb.py              # 首次构建
  python build_kb.py --rebuild    # 强制重建
"""

import os, sys, json, argparse, time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(PROJECT_DIR, "kb")

# ── Embedding 模型 (可通过环境变量覆盖) ──
EMBED_MODEL = os.environ.get(
    "MAZU_EMBED_MODEL",
    "BAAI/bge-small-zh-v1.5",  # 中文优化, 24MB, 384维
    # 备选: "sentence-transformers/all-MiniLM-L6-v2" (多语言, 23MB)
    # 备选: "shibing624/text2vec-base-chinese" (中文, 130MB)
)


# ── 简单中文文本分块器（纯 Python，零依赖） ──

SEPARATORS = [
    "\n\n",     # Markdown 段落
    "\n",       # 单行换行
    "。",       # 中文句号
    "；",       # 中文分号
    "，",       # 中文逗号
    " ",        # 空格
]


def _split_text(text: str, chunk_size: int = 600) -> list:
    """将长文本分割为不超过 chunk_size 的块（迭代实现）。"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    result = []

    # 找到最佳切割点: 在 chunk_size 范围内找最后一个分隔符
    def _best_split(s: str) -> int:
        if len(s) <= chunk_size:
            return len(s)
        # 在 [chunk_size//2, chunk_size] 范围内找最佳断点
        window_start = max(chunk_size // 2, 0)
        window = s[window_start:chunk_size]
        best = chunk_size
        for sep in SEPARATORS:
            idx = window.rfind(sep)
            if idx >= 0:
                pos = window_start + idx + len(sep)
                if chunk_size // 2 < pos <= chunk_size:
                    best = min(best, pos)
        return best

    remaining = text
    while len(remaining) > chunk_size:
        cut = _best_split(remaining)
        if cut <= 0:
            cut = chunk_size  # 兜底
        result.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()

    if remaining.strip():
        result.append(remaining)

    return result


def chunk_documents(documents: list, chunk_size: int = 600, overlap: int = 80) -> list:
    """将 Document 列表分块，返回新的 Document 列表。

    每个块带有 overlap 字符的前后文衔接。
    """
    from langchain_core.documents import Document

    chunks = []
    for doc in documents:
        text = doc.page_content
        if len(text) <= chunk_size:
            chunks.append(doc)
            continue

        raw_chunks = _split_text(text, chunk_size)

        for i, raw in enumerate(raw_chunks):
            # 前一块尾部作为前缀overlap
            if i > 0:
                prev = raw_chunks[i-1]
                prefix = prev[-overlap:] if len(prev) > overlap else prev
                raw = prefix + "\n" + raw

            chunks.append(Document(
                page_content=raw.strip(),
                metadata=dict(doc.metadata),
            ))

    return chunks


def get_embeddings():
    """延迟加载 embedding 模型（避免启动时下载）。"""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # model_kwargs 传递给 sentence-transformers
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ══════════════════════════════════════════════════════════════
# Document Loaders
# ══════════════════════════════════════════════════════════════

def load_docs() -> list:
    """加载 docs/ 目录下所有 Markdown 文档。

    返回 langchain Document 列表，metadata 记录文件名和类型。
    """
    from langchain_core.documents import Document

    docs_dir = os.path.join(PROJECT_DIR, "docs")
    if not os.path.isdir(docs_dir):
        print(f"⚠ docs/ 目录不存在，跳过文档加载")
        return []

    documents = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(docs_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            continue

        # 用文件名作为标题元数据
        title = fname.replace(".md", "").replace("_", " ").replace("-", " ")
        documents.append(Document(
            page_content=content,
            metadata={
                "source": f"docs/{fname}",
                "type": "document",
                "title": title,
            },
        ))
        print(f"  [DOC] docs/{fname}  ({len(content)} 字符)")

    return documents


def load_indicators() -> list:
    """从 schema/operators.json 加载指标定义。

    每个指标生成一个独立 Document，包含完整定义、公式和局限性。
    """
    from langchain_core.documents import Document

    ops_path = os.path.join(PROJECT_DIR, "schema", "operators.json")
    if not os.path.exists(ops_path):
        print(f"⚠ {ops_path} 不存在，跳过指标加载")
        return []

    with open(ops_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for op in data.get("operators", []):
        ind_id = op.get("id", "?")
        category = op.get("category", "")

        lines = [f"## 气象指标: {ind_id}"]
        lines.append(f"分类: {category}")
        lines.append(f"描述: {op.get('description', '无')}")

        if op.get("expression"):
            lines.append(f"计算公式: {op['expression']}")
        if op.get("output_unit"):
            lines.append(f"输出单位: {op['output_unit']}")
        if op.get("source"):
            lines.append(f"数据来源: {op['source']}")
        if op.get("co_occurs_with"):
            lines.append(f"联合解释指标: {', '.join(op['co_occurs_with'])}")
        if op.get("limitations"):
            lines.append(f"局限性说明: {op['limitations']}")
        if op.get("inputs"):
            lines.append(f"输入变量: {json.dumps(op['inputs'], ensure_ascii=False)}")
        if op.get("dag"):
            lines.append(f"推导步骤(DAG): {json.dumps(op['dag'], ensure_ascii=False)}")

        text = "\n".join(lines)
        documents.append(Document(
            page_content=text,
            metadata={
                "source": "schema/operators.json",
                "type": "indicator",
                "indicator_id": ind_id,
                "category": category,
            },
        ))

    print(f"  [DATA] schema/operators.json → {len(documents)} 个指标")
    return documents


def load_rules() -> list:
    """从 schema/rules.json 加载检测规则。

    每条规则 + 每个条件生成独立块，方便精确检索。
    """
    from langchain_core.documents import Document

    rules_path = os.path.join(PROJECT_DIR, "schema", "rules.json")
    if not os.path.exists(rules_path):
        print(f"⚠ {rules_path} 不存在，跳过规则加载")
        return []

    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for rule in data.get("rules", []):
        htype = rule.get("hazard_type", "?")
        htype_cn = {
            "flash_flood": "山洪",
            "extreme_heat": "极端高温",
            "dust_storm": "沙尘强风",
            "coastal_humid_heat": "沿海湿热",
        }.get(htype, htype)

        # ── 规则整体 ──
        rule_text = f"## 检测规则: {htype_cn} ({htype})\n"
        rule_text += f"严重度等级: {json.dumps(rule.get('severity', {}), ensure_ascii=False)}\n"
        if rule.get("fallback"):
            rule_text += f"降级策略: {json.dumps(rule['fallback'], ensure_ascii=False)}\n"

        # ── 每个条件独立一块 (精确检索) ──
        for i, cond in enumerate(rule.get("conditions", [])):
            ind_id = cond.get("indicator", "?")
            role = cond.get("role", cond.get("type", ""))
            lines = [
                f"## 检测条件: {htype_cn} → 条件 #{i+1}",
                f"灾害类型: {htype_cn} ({htype})",
                f"指标: {ind_id}",
                f"条件: {cond.get('op', cond.get('condition', '>='))} {cond.get('value', '?')}",
                f"权重: {cond.get('weight', '?')}",
                f"因果角色: {role}",
            ]
            if cond.get("primary"):
                lines.append("角色: 主门控 (primary gate) — 不通过则评分×0.25")
            if cond.get("description"):
                lines.append(f"说明: {cond['description']}")

            documents.append(Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "schema/rules.json",
                    "type": "rule_condition",
                    "hazard_type": htype,
                    "indicator_id": ind_id,
                    "role": role,
                },
            ))

        # 规则整体作为一块 (概括性查询)
        if rule.get("conditions"):
            rule_text += f"\n条件数量: {len(rule['conditions'])}"
            rule_text += f"\n条件指标: {', '.join(c['indicator'] for c in rule['conditions'])}"
            documents.append(Document(
                page_content=rule_text,
                metadata={
                    "source": "schema/rules.json",
                    "type": "rule_overview",
                    "hazard_type": htype,
                },
            ))

    print(f"  [RULE] schema/rules.json → {len(documents)} 个文档块 ({len(data.get('rules',[]))} 条规则)")
    return documents


def load_agent_knowledge() -> list:
    """从 agent.py 系统 Prompt 提取沙特气候研判知识。

    只提取知识部分（能力边界、研判规则、沙特地理），
    跳过工具列表和输出规范（那些是给 LLM 的指令，不是领域知识）。
    """
    from langchain_core.documents import Document

    agent_path = os.path.join(PROJECT_DIR, "agent.py")
    if not os.path.exists(agent_path):
        print(f"⚠ agent.py 不存在，跳过知识提取")
        return []

    with open(agent_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取 _build_system_prompt() 函数体 (三引号之间的大段文本)
    # 简单策略：提取 return f""" ... """ 的内容
    import re
    match = re.search(r'return f"""(.*?)"""', content, re.DOTALL)
    if not match:
        print("⚠ 未找到系统 Prompt 文本，跳过知识提取")
        return []

    prompt_text = match.group(1)

    # 按 "═══════" 分割成段
    sections = re.split(r'═{10,}', prompt_text)

    documents = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        # 给每个段落一个标题
        lines = section.strip().split("\n")
        title = lines[0].strip() if lines else ""
        if not title:
            continue

        # 过滤纯指令性内容 (输出规范/禁止行为是给 LLM 的 prompt 指令)
        if any(kw in title for kw in ["输出规范", "禁止行为", "调用对应工具"]):
            continue

        documents.append(Document(
            page_content=section.strip(),
            metadata={
                "source": "agent.py (system prompt)",
                "type": "climate_knowledge",
                "section": title,
            },
        ))

    print(f"  [PROMPT] agent.py 系统 Prompt → {len(documents)} 个知识块")
    return documents


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def build_kb(rebuild: bool = False):
    """主入口：加载所有文档 → 分块 → 向量化 → 存入 ChromaDB。

    Args:
        rebuild: True 时删除已有 kb/ 目录完全重建。
    """
    import shutil

    if rebuild and os.path.exists(KB_DIR):
        shutil.rmtree(KB_DIR)
        print(f"[DEL]  已删除旧知识库: {KB_DIR}\n")

    if os.path.exists(KB_DIR) and os.listdir(KB_DIR):
        print(f"[OK] 知识库已存在: {KB_DIR}")
        print("   如需重建，运行: python build_kb.py --rebuild")
        return

    t0 = time.time()

    # ── 1. 加载原始文档 ──
    print("=" * 55)
    print("[LOAD] 加载文档…")
    print("=" * 55)

    raw_docs = []
    raw_docs.extend(load_docs())
    raw_docs.extend(load_indicators())
    raw_docs.extend(load_rules())
    raw_docs.extend(load_agent_knowledge())

    print(f"\n[FILE] 原始文档总数: {len(raw_docs)}")

    if not raw_docs:
        print("[ERR] 没有找到任何文档，退出。")
        return

    # ── 2. 分块 ──
    print(f"\n{'='*55}")
    print("[CHUNK] 文本分块...")
    print(f"{'='*55}")

    chunks = chunk_documents(raw_docs)

    print(f"  分块完成: {len(raw_docs)} 个原始文档 → {len(chunks)} 个块")
    print(f"  平均块大小: {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} 字符")

    # ── 3. 向量化 + 存储 ──
    print(f"\n{'='*55}")
    print("[EMBED] 向量化 + 存储到 ChromaDB…")
    print(f"   Embedding 模型: {EMBED_MODEL}")
    print(f"{'='*55}")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=KB_DIR,
        collection_name="mazu_knowledge",
    )
    # Chroma.from_documents 在最新版本自动 persist，
    # 但显式调用确保落盘
    if hasattr(vectorstore, "_persist"):
        vectorstore._persist()

    elapsed = time.time() - t0
    print(f"\n[OK] 知识库构建完成！")
    print(f"   位置: {KB_DIR}")
    print(f"   块数: {len(chunks)}")
    print(f"   耗时: {elapsed:.1f}s")

    # ── 4. 快速检索测试 ──
    print(f"\n{'='*55}")
    print("[TEST] 检索测试…")
    print(f"{'='*55}")

    test_queries = [
        "极端高温的检测条件是什么？",
        "沙尘暴和 Shamal 风有什么关系？",
        "红海沿岸有哪些山洪风险？",
        "t2m_c 指标怎么计算的？",
    ]

    for q in test_queries:
        results = vectorstore.similarity_search(q, k=2)
        sources = set(doc.metadata.get("source", "?") for doc in results)
        print(f"  [Q] {q}")
        print(f"     → {len(results)} 个结果, 来源: {', '.join(sources)}")


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAZU 知识库构建工具")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="删除已有知识库后重建",
    )
    args = parser.parse_args()

    # 延迟导入 Chroma
    from langchain_community.vectorstores import Chroma

    build_kb(rebuild=args.rebuild)
