# Agent/Agent_Config/agent_config.py
import os
import datetime
from rich.console import Console

# ============ Rich Console ============
console = Console()

# The complete corpus is stored outside the source repository. Override this
# location with BEAVER_CORPUS_ROOT before running the Agent.
_DEFAULT_CORPUS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "example_corpus",
)
CORPUS_ROOT = os.path.abspath(
    os.path.expanduser(os.getenv("BEAVER_CORPUS_ROOT", _DEFAULT_CORPUS_ROOT))
)


def _corpus_path(*parts):
    return os.path.join(CORPUS_ROOT, *parts)


BM25_CACHE_DIR = _corpus_path("BM25_CACHE")
TOKENIZER_TAG = "en_regex_v1"  # 用于区分缓存，换分词逻辑时改这个

# ============ 多语料库配置 ============
CHUNK_CORPUS_CONFIG = [
    {
        "name": "RSC",
        "embed_path": _corpus_path("RSC", "Embeddings_label", "file_embeddings.npy"),
        "meta_path": _corpus_path("RSC", "Embeddings_label", "file_metadata_cleaned.csv"),
        "granularity": "chunk",
    },
    {
        "name": "Wiley",
        "embed_path": _corpus_path("Wiley", "Embeddings_label", "file_embeddings.npy"),
        "meta_path": _corpus_path("Wiley", "Embeddings_label", "file_metadata_cleaned.csv"),
        "granularity": "chunk",
    },
    {
        "name": "AIP",
        "embed_path": _corpus_path("AIP", "Embeddings_label", "file_embeddings.npy"),
        "meta_path": _corpus_path("AIP", "Embeddings_label", "file_metadata_cleaned.csv"),
        "granularity": "chunk",
    },
    {
        "name": "IOP",
        "embed_path": _corpus_path("IOP", "Embeddings_label", "file_embeddings.npy"),
        "meta_path": _corpus_path("IOP", "Embeddings_label", "file_metadata_cleaned.csv"),
        "granularity": "chunk",
    },
    {
        "name": "Elsevier",
        "embed_path": _corpus_path("Elsevier", "Embeddings_label", "file_embeddings.cleaned.npy"),
        "meta_path": _corpus_path("Elsevier", "Embeddings_label", "file_metadata.cleaned_cleaned.csv"),
        "granularity": "chunk",
    },
    {
        "name": "Springer",
        "embed_path": _corpus_path("Springer", "Embeddings_label", "file_embeddings.npy"),
        "meta_path": _corpus_path("Springer", "Embeddings_label", "file_metadata_cleaned.csv"),
        "granularity": "chunk",
    },
    {
        "name": "ACS",
        "embed_path": _corpus_path("ACS", "Embeddings_label", "file_embeddings.cleaned.npy"),
        "meta_path": _corpus_path("ACS", "Embeddings_label", "file_metadata.cleaned_cleaned.csv"),
        "granularity": "chunk",
    },
]

PARA_CORPUS_CONFIG = [
    {
        "name": "RSC",
        "embed_path": _corpus_path("RSC", "PARA_CORPUS", "file_embeddings.npy"),
        "meta_path": _corpus_path("RSC", "PARA_CORPUS", "file_metadata.csv"),
        "granularity": "paras",
    },
    {
        "name": "Wiley",
        "embed_path": _corpus_path("Wiley", "PARA_CORPUS", "file_embeddings.npy"),
        "meta_path": _corpus_path("Wiley", "PARA_CORPUS", "file_metadata.csv"),
        "granularity": "paras",
    },
    {
        "name": "AIP",
        "embed_path": _corpus_path("AIP", "PARA_CORPUS", "file_embeddings.npy"),
        "meta_path": _corpus_path("AIP", "PARA_CORPUS", "file_metadata.csv"),
        "granularity": "paras",
    },
    {
        "name": "IOP",
        "embed_path": _corpus_path("IOP", "PARA_CORPUS", "file_embeddings.npy"),
        "meta_path": _corpus_path("IOP", "PARA_CORPUS", "file_metadata.csv"),
        "granularity": "paras",
    },
    {
        "name": "Elsevier",
        "embed_path": _corpus_path("Elsevier", "PARA_CORPUS", "file_embeddings.cleaned.npy"),
        "meta_path": _corpus_path("Elsevier", "PARA_CORPUS", "file_metadata.cleaned.csv"),
        "granularity": "paras",
    },
    {
        "name": "Springer",
        "embed_path": _corpus_path("Springer", "PARA_CORPUS", "file_embeddings.npy"),
        "meta_path": _corpus_path("Springer", "PARA_CORPUS", "file_metadata.csv"),
        "granularity": "paras",
    },
    {
        "name": "ACS",
        "embed_path": _corpus_path("ACS", "PARA_CORPUS", "file_embeddings.cleaned.npy"),
        "meta_path": _corpus_path("ACS", "PARA_CORPUS", "file_metadata.cleaned.csv"),
        "granularity": "paras",
    },
]

CORPUS_GRANULARITY = os.getenv(
    "BEAVER_CORPUS_GRANULARITY", "chunk"
).strip().lower()

if CORPUS_GRANULARITY == "chunk":
    CORPUS_CONFIG = CHUNK_CORPUS_CONFIG
elif CORPUS_GRANULARITY in {"para", "paras", "paragraph", "paragraphs"}:
    CORPUS_CONFIG = PARA_CORPUS_CONFIG
else:
    raise ValueError(
        "BEAVER_CORPUS_GRANULARITY must be 'chunk' or 'paras', "
        f"not {CORPUS_GRANULARITY!r}."
    )

# ============ 📂 自动生成动态日志路径 ============
# 1. 你的基础目录
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_ROOT = os.path.dirname(AGENT_ROOT)
BASE_OUTPUT_FOLDER = os.path.join(AGENT_ROOT, "Session_Runs")

# 🟢 兼容旧代码变量名
OUTPUT_FOLDER = BASE_OUTPUT_FOLDER

# 2. 自动生成时间戳
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. 拼接最终路径
STEP_LOG_DIR = os.path.join(BASE_OUTPUT_FOLDER, f"Run_{TIMESTAMP}")

# 4. 自动创建文件夹
os.makedirs(STEP_LOG_DIR, exist_ok=True)

# ============ 数据库路径 ============
MECH_DB_PATH = os.path.join(
    SUBMISSION_ROOT,
    "ML_Tool",
    "3.ML_Tool",
    "Prediction-AB.csv",
)


# ============ 🤖 模型名称配置 ============
EMBED_MODEL = "GLM-Embedding-2"

# --- 核心模型定义 (API Model Names) ---
# [Main] 推理模型 (DeepSeek R1): 负责 Planner, Executor 等核心逻辑
LLM_MODEL = "DeepSeek-R1"

# [Fast] 快速模型 (DeepSeek V3): 负责 Intent, Report, Score 以及 Benchmark 对照组
LLM_MODEL_V3 =  "DeepSeek-V3.2"

# --- 模块功能分配 ---
INTENT_MODEL         = LLM_MODEL_V3  # 意图识别 (速度优先)
DESIGN_MODEL         = LLM_MODEL     # 实验设计 (深度推理)
PLANNER_Module_MODEL = LLM_MODEL     # 任务拆解 (深度推理)
SCORE_MODEL          = LLM_MODEL_V3  # Rerank打分 (速度与遵循指令)
REPORT_MODEL         = LLM_MODEL_V3  # 论文写作 (V3 写文采很好且快)
LLM_MODEL_RAGAS      = LLM_MODEL_V3  # RAGAS 评测 (作为裁判)

# MODEL_NAME = "gemini-3-pro-preview-c"

print(f"🔧 [Config] Log Directory initialized at: {STEP_LOG_DIR}")
print(f"🔧 [Config] Main LLM: {LLM_MODEL} | Fast LLM: {LLM_MODEL_V3}")
