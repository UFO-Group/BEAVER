import os
import sys
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from rank_bm25 import BM25Okapi
from numpy.lib.format import open_memmap

# =========================
# ✅ 项目路径注入
# =========================
current_file = os.path.abspath(__file__)
rag_dir = os.path.dirname(current_file)      # .../Agent/RAG
agent_dir = os.path.dirname(rag_dir)         # .../Agent
project_root = os.path.dirname(agent_dir)    # .../ProjectRoot

if project_root not in sys.path:
    sys.path.append(project_root)

from Agent.Agent_Config.agent_config import BM25_CACHE_DIR, CORPUS_CONFIG, console

# =========================
# ✅ BM25 缓存配置（你想放哪就改哪）
# =========================
DEFAULT_BM25_CACHE_DIR = str(BM25_CACHE_DIR)
TOKENIZER_TAG = "en_regex_v1"  # 以后你换分词逻辑，就改这个，缓存自动隔离

# =========================
# ✅ 全局变量（其他模块会用）
# =========================
# ✅ Dense shard cache
shards = []  # 每个元素: {"name","emb","norms","offset","n"}
embeddings = None
metadata = None
bm25_model = None

# ✅ [新增-2] 记录当前 bm25_model 对应的缓存文件路径，避免切换配置时串索引
bm25_cache_path_loaded = None

# =========================
# ✅ joblib / pickle 兜底
# =========================
try:
    import joblib

    def dump_obj(obj, path: str):
        joblib.dump(obj, path, compress=3)

    def load_obj(path: str):
        return joblib.load(path)

except ImportError:
    import pickle

    def dump_obj(obj, path: str):
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_obj(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

# =========================
# ✅ 英文 tokenizer（regex）
# =========================
_token_pat = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[A-Za-z0-9\-_]+")

def tokenize_en(s: str):
    return _token_pat.findall(str(s).lower())

# =========================
# ✅ CSV 读取：编码兜底 + 坏行兜底
# =========================
def read_csv_fallback(path: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "cp1252", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except ParserError:
            # 字段数不一致/坏行：用 python 引擎兜底
            try:
                return pd.read_csv(
                    path,
                    encoding=enc,
                    engine="python",
                    sep=None,              # 自动识别分隔符
                    on_bad_lines="warn",   # 也可改成 "skip" 更干净但会丢行
                    **kwargs
                )
            except Exception as e:
                last_err = e
                continue

    # pandas>=2.0: encoding_errors
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except TypeError:
        # 老 pandas：latin1 基本必能读
        return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="warn", **kwargs)
    except Exception as e:
        raise RuntimeError(f"CSV 读取失败: {path}，最后错误: {last_err}") from e

def ensure_norms_cache(emb: np.ndarray, norms_path: str, block: int = 20000) -> np.ndarray:
    """
    给一个 embeddings(memmap 或 ndarray) 生成/加载 norms（float32, shape=(N,)）
    norms 用于精确 cosine: cos = (X@q) / (||X||*||q||)
    """
    if os.path.exists(norms_path):
        return np.load(norms_path, mmap_mode="r")

    console.print(f"🔧 构建 norms 缓存：{norms_path}")
    Path(os.path.dirname(norms_path)).mkdir(parents=True, exist_ok=True)

    N = emb.shape[0]
    norms_mm = open_memmap(norms_path, mode="w+", dtype=np.float32, shape=(N,))

    for i in range(0, N, block):
        Xi = emb[i:i+block]
        # ✅ einsum 不创建 Xi*Xi 大临时矩阵
        row_sq = np.einsum("ij,ij->i", Xi, Xi)
        norms_mm[i:i+block] = (np.sqrt(row_sq).astype(np.float32) + 1e-12)

    norms_mm.flush()
    return np.load(norms_path, mmap_mode="r")
# =========================
# ✅ 语料签名：语料变化 => 缓存自动失效（换新 BM 文件）
# =========================
def corpus_signature(corpus_config) -> str:
    info = []
    for cfg in corpus_config:
        meta_path = cfg["meta_path"]
        st = os.stat(meta_path)
        info.append({
            "path": str(meta_path),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        })
    s = json.dumps(info, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def bm25_cache_path(cache_dir: str, granularity: str, text_col: str, tokenizer_tag: str, sig: str) -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fname = f"bm25_{granularity}_{text_col}_{tokenizer_tag}_{sig}.joblib"
    return str(Path(cache_dir) / fname)

# =========================
# ✅ 找文本列（你可以按实际 CSV 列名增补）
# =========================
def pick_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "text", "content", "chunk", "paragraph", "para",
        "clean_text", "raw_text", "chunk_text"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"metadata 中找不到文本列，现有列: {list(df.columns)}")

# =========================
# ✅ 内部：只构建/加载 BM25（不重载 embeddings/metadata）
# =========================
def _ensure_bm25(
    bm25_cache_dir: str,
    bm25_text_col: str = "text_for_emb",
    bm25_max_docs: int | None = None,
    granularity_key: str = "chunk",
):
    global bm25_model, bm25_cache_path_loaded, metadata

    if metadata is None:
        raise RuntimeError("metadata is None，无法构建 BM25。请先加载向量库/metadata。")

    if bm25_text_col is None:
        bm25_text_col = pick_text_col(metadata)

    if bm25_text_col not in metadata.columns:
        raise KeyError(f"❌ metadata 中没有列 {bm25_text_col}，现有列：{list(metadata.columns)}")

    sig = corpus_signature(CORPUS_CONFIG)
    cache_path = bm25_cache_path(
        cache_dir=bm25_cache_dir,
        granularity=granularity_key,   # ✅ 不要写死 chunk
        text_col=bm25_text_col,
        tokenizer_tag=TOKENIZER_TAG,
        sig=sig
    )

    # ✅ [新增-2] 如果 bm25_model 已存在，但缓存路径不一致（粒度/列/语料签名变了），必须切换
    if bm25_model is not None and bm25_cache_path_loaded != cache_path:
        console.print(f"🔁 BM25 配置变化，切换索引：\n  - old: {bm25_cache_path_loaded}\n  - new: {cache_path}")
        bm25_model = None
        bm25_cache_path_loaded = None

    # 已是目标索引
    if bm25_model is not None and bm25_cache_path_loaded == cache_path:
        console.print("✅ BM25 已在内存中（匹配当前配置），跳过加载/构建")
        return

    # 尝试加载缓存
    if os.path.exists(cache_path):
        console.print(f"✅ 发现 BM25 缓存，直接加载：{cache_path}")
        bm25_model = load_obj(cache_path)
        bm25_cache_path_loaded = cache_path
        console.print("✅ BM25 缓存加载完成！")
        return

    # 构建并保存
    console.print(
        f"📚 未发现缓存，正在构建 BM25（{bm25_text_col} / docs={len(metadata)}）..."
    )
    corpus_text = metadata[bm25_text_col].fillna("").astype(str).tolist()

    if bm25_max_docs is not None:
        corpus_text = corpus_text[:bm25_max_docs]
        console.print(f"⚠️ bm25_max_docs={bm25_max_docs}：仅用前 {len(corpus_text)} 条构建 BM25（调试模式）")

    tokenized_corpus = [tokenize_en(doc) for doc in corpus_text]
    bm25_model = BM25Okapi(tokenized_corpus)

    console.print(f"💾 保存 BM25 缓存到：{cache_path}")
    dump_obj(bm25_model, cache_path)
    bm25_cache_path_loaded = cache_path
    console.print("✅ BM25 索引构建并缓存完成！")

# =========================
# ✅ 主函数：加载向量 + meta，并构建/加载 BM25（带缓存）
# ✅ 新增：构建 shards + norms（用于“分库并行精确 topK + 全局 merge”）
# =========================
def load_vector_store(
    build_bm25: bool = True,
    bm25_cache_dir: str = DEFAULT_BM25_CACHE_DIR,
    bm25_text_col: str | None = "text_for_emb",
    bm25_max_docs: int | None = None,
):
    """
    输出到本模块全局变量：
      - embeddings: combined memmap (float32)  [可选保留，兼容旧逻辑]
      - metadata:   全库合并后的 metadata (DataFrame)
      - shards:     分库切片列表，用于“并行精确检索”
      - emb_norms:  全库 norms memmap，用于精确 cosine

    shards[i] 结构：
      {
        "name": "RSC",
        "granularity": "chunk",
        "emb": <memmap slice (n,d)>,
        "norms": <memmap slice (n,)>,
        "offset": start_row_in_global,
        "n": n_rows
      }
    """
    global embeddings, metadata, bm25_model

    # ✅ 新增全局：shards + emb_norms
    global shards, emb_norms
    try:
        shards
    except NameError:
        shards = []
    try:
        emb_norms
    except NameError:
        emb_norms = None

    from pathlib import Path
    from numpy.lib.format import open_memmap
    import numpy as np
    import pandas as pd
    import os

    all_meta = []
    emb_blocks = []      # (emb_path, nrows, dim)
    shard_specs = []     # (name, granularity, start, end)

# 1) 先读 metadata + 统计总行数 + 记录每库的全局 offset
    sig = corpus_signature(CORPUS_CONFIG)
    Path(bm25_cache_dir).mkdir(parents=True, exist_ok=True)
    meta_cache_path = os.path.join(bm25_cache_dir, f"combined_metadata_{sig}.parquet")
    
    total_rows = 0
    dim = None
    offset = 0

    if os.path.exists(meta_cache_path):
        console.print(f"⚡ 发现 Metadata 高速缓存，极速加载: {meta_cache_path}")
        metadata = pd.read_parquet(meta_cache_path)
        total_rows = len(metadata)
        
        for cfg in CORPUS_CONFIG:
            emb_path = cfg["embed_path"]
            emb = np.load(emb_path, mmap_mode="r")
            if dim is None: dim = emb.shape[1]
            nrows = emb.shape[0]
            emb_blocks.append((emb_path, nrows, dim))
            shard_specs.append((cfg["name"], cfg.get("granularity", "chunk"), offset, offset + nrows))
            offset += nrows
    else:
        for cfg in CORPUS_CONFIG:
            console.print(f"\n📂 加载语料库: [bold]{cfg['name']}[/bold]")
            emb_path = cfg["embed_path"]
            meta_path = cfg["meta_path"]
            granularity = cfg.get("granularity", "chunk")

            meta = read_csv_fallback(meta_path)
            meta["corpus"] = cfg["name"]
            meta["granularity"] = granularity
            all_meta.append(meta)

            emb = np.load(emb_path, mmap_mode="r")
            if dim is None: dim = emb.shape[1]
            nrows = emb.shape[0]
            
            emb_blocks.append((emb_path, nrows, dim))
            shard_specs.append((cfg["name"], granularity, offset, offset + nrows))
            offset += nrows
            total_rows += nrows

        metadata = pd.concat(all_meta, ignore_index=True)
        metadata.to_parquet(meta_cache_path, index=False)
        console.print(f"💾 Metadata 已保存至高速缓存！")

    # 2) 拼接成 combined embeddings memmap（只做一次；之后复用）
    sig = corpus_signature(CORPUS_CONFIG)
    Path(bm25_cache_dir).mkdir(parents=True, exist_ok=True)

    combined_path = os.path.join(bm25_cache_dir, f"combined_embeddings_{sig}_f32.npy")

    if os.path.exists(combined_path):
        console.print(f"✅ 发现组合向量库缓存，直接加载: {combined_path}")
        embeddings = np.load(combined_path, mmap_mode="r")
    else:
        console.print(f"📦 正在构建组合向量库（memmap）: {combined_path}")
        mm = open_memmap(combined_path, mode="w+", dtype=np.float32, shape=(total_rows, dim))

        off = 0
        for emb_path, nrows, _ in emb_blocks:
            emb = np.load(emb_path, mmap_mode="r")
            mm[off:off+nrows] = emb.astype(np.float32, copy=False)
            off += nrows

        mm.flush()
        embeddings = np.load(combined_path, mmap_mode="r")

    console.print(
        f"\n✅ 向量总数: [bold green]{embeddings.shape[0]}[/bold green]，维度: [bold]{embeddings.shape[1]}[/bold]"
    )

    # 3) ✅ 新增：构建/加载 norms（精确 cosine 用）
    norms_dir = os.path.join(bm25_cache_dir, "NORMS_CACHE")
    Path(norms_dir).mkdir(parents=True, exist_ok=True)
    norms_path = os.path.join(norms_dir, f"combined_norms_{sig}_f32.npy")

    if os.path.exists(norms_path):
        console.print(f"✅ 发现 norms 缓存，直接加载: {norms_path}")
        emb_norms = np.load(norms_path, mmap_mode="r")
    else:
        console.print(f"🔧 未发现 norms 缓存，开始构建: {norms_path}")
        norms_mm = open_memmap(norms_path, mode="w+", dtype=np.float32, shape=(total_rows,))

        block = 20000
        X = embeddings  # memmap float32
        for i in range(0, total_rows, block):
            Xi = X[i:i+block]
            # ✅ einsum 不会生成 Xi*Xi 的巨型临时数组
            row_sq = np.einsum("ij,ij->i", Xi, Xi)
            norms_mm[i:i+block] = (np.sqrt(row_sq).astype(np.float32) + 1e-12)

        norms_mm.flush()
        emb_norms = np.load(norms_path, mmap_mode="r")

    # 4) ✅ 新增：构建 shards（分库切片，供 rag_core 并行精确检索）
    shards = []
    for name, granularity, start, end in shard_specs:
        shards.append({
            "name": name,
            "granularity": granularity,
            "emb": embeddings[start:end],      # memmap slice (n,d)
            "norms": emb_norms[start:end],     # memmap slice (n,)
            "offset": start,
            "n": end - start,
        })

    console.print(f"✅ shards 构建完成：{len(shards)} 个")
    for s in shards:
        console.print(f"   - {s['name']}: n={s['n']} | offset={s['offset']} | granularity={s['granularity']}")

    # 5) 可选：构建/加载 BM25
    if not build_bm25:
        console.print("⏭️ 跳过 BM25 构建（build_bm25=False）")
        return

    _ensure_bm25(
        bm25_cache_dir=bm25_cache_dir,
        bm25_text_col="text_for_emb",
        bm25_max_docs=bm25_max_docs,
        granularity_key="+".join(sorted(metadata["granularity"].unique()))
    )
