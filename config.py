"""
全局配置
========
- 路径、模型、体验旋钮维度、形态要素维度
- 从环境变量 / .env 读取 API Key；无密钥时自动 MOCK
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env_path(key: str, default: Path) -> Path:
    """读取环境变量路径；不存在或无效时回退 default。"""
    raw = os.getenv(key, "").strip().strip('"').strip("'")
    if not raw:
        return default
    p = Path(raw).expanduser()
    try:
        p = p.resolve()
    except OSError:
        return default
    return p if p.exists() else default


# ---------- 路径 ----------
# 同时兼容 main 的 DATA_DIR 与 Task 2/3 分支的 AI4CITY_DATA_DIR；外部数据只读。
_default_data_dir = ROOT.parent / "ai4city-data"
DATA_DIR = _env_path(
    "AI4CITY_DATA_DIR",
    _env_path("DATA_DIR", _default_data_dir),
)

KB_DIR = ROOT / "knowledge_base" / "data"
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_OUT_DIR = OUTPUT_DIR / "images"
SESSION_DIR = OUTPUT_DIR / "sessions"
PANORAMA_VIEW_CACHE_DIR = Path(
    os.getenv("PANORAMA_VIEW_CACHE_DIR", str(OUTPUT_DIR / "panorama_views"))
).resolve()
RAG_CACHE_DIR = Path(
    os.getenv("RAG_CACHE_DIR", str(OUTPUT_DIR / "rag_cache"))
).resolve()
KNOWLEDGE_DRAFT_DIR = Path(
    os.getenv("KNOWLEDGE_DRAFT_DIR", str(OUTPUT_DIR / "knowledge_drafts"))
).resolve()
RAG_KNOWLEDGE_DIR = Path(
    os.getenv("RAG_KNOWLEDGE_DIR", str(ROOT / "rag_knowledge"))
).resolve()
RAG_PUBLISHED_KNOWLEDGE_DIR = RAG_KNOWLEDGE_DIR / "published"

# 原图目录：优先 LOAD_DATA_DIR / DATA_DIR/assets / ROOT/assets
_load_assets = os.getenv("LOAD_DATA_DIR", "").strip().strip('"').strip("'")
if _load_assets:
    _candidate = (ROOT / _load_assets).resolve() if not Path(_load_assets).is_absolute() else Path(_load_assets)
    ASSETS_DIR = _candidate if _candidate.exists() else (DATA_DIR / "assets")
else:
    ASSETS_DIR = DATA_DIR / "assets" if (DATA_DIR / "assets").exists() else ROOT / "assets"

# 派生图、生成结果和缓存只写仓库输出目录，不写外部数据目录。
TARGET_IMG_DIR = ROOT / "TargetIMG"

# 指标表与三张分析图目录
FILLED_METRICS_XLSX = (
    DATA_DIR / "filled_metrics.xlsx"
    if (DATA_DIR / "filled_metrics.xlsx").exists()
    else ROOT / "filled_metrics.xlsx"
)
EDGE_MAPS_DIR = (
    DATA_DIR / "edge_density_maps"
    if (DATA_DIR / "edge_density_maps").exists()
    else ROOT / "edge_density_maps"
)
SEG_MAPS_DIR = (
    DATA_DIR / "segmentation_results"
    if (DATA_DIR / "segmentation_results").exists()
    else ROOT / "segmentation_results"
)
SKYLINE_MAPS_DIR = (
    DATA_DIR / "skyline_boundary_maps"
    if (DATA_DIR / "skyline_boundary_maps").exists()
    else ROOT / "skyline_boundary_maps"
)

# Excel B 列 / 文件名匹配用的固定前缀长度
IMAGE_KEY_LEN = 26

# Task 2/3 后端目录与场景分类。
PANORAMA_DIR = Path(
    os.getenv("AI4CITY_PANORAMA_DIR", str(ASSETS_DIR))
).resolve()
METRICS_TABLE_DIR = Path(
    os.getenv("AI4CITY_METRICS_TABLE_DIR", str(DATA_DIR))
).resolve()
SCENE_MANIFEST_PATH = Path(
    os.getenv("AI4CITY_SCENE_MANIFEST", str(DATA_DIR / "scenes.csv"))
).resolve()
SCENE_TYPES_ZH = ("社区", "蓝绿", "商办")
_metrics_table_path_raw = os.getenv("AI4CITY_METRICS_TABLE", "").strip()
METRICS_TABLE_PATH = (
    Path(_metrics_table_path_raw).resolve()
    if _metrics_table_path_raw
    else FILLED_METRICS_XLSX if FILLED_METRICS_XLSX.is_file() else None
)
KNOWLEDGE_SOURCE_DIR = Path(
    os.getenv("AI4CITY_KNOWLEDGE_DIR", str(DATA_DIR / "knowledge"))
).resolve()

for _p in (
    KB_DIR,
    UPLOAD_DIR,
    IMAGE_OUT_DIR,
    SESSION_DIR,
    PANORAMA_VIEW_CACHE_DIR,
    RAG_CACHE_DIR,
    KNOWLEDGE_DRAFT_DIR,
    RAG_PUBLISHED_KNOWLEDGE_DIR,
    TARGET_IMG_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

# ---------- Task 2/3 全景视图与场景理解 ----------
PANORAMA_OVERVIEW_WIDTH = int(os.getenv("PANORAMA_OVERVIEW_WIDTH", "2048"))
PANORAMA_OVERVIEW_HEIGHT = int(os.getenv("PANORAMA_OVERVIEW_HEIGHT", "1024"))
PANORAMA_PERSPECTIVE_WIDTH = int(os.getenv("PANORAMA_PERSPECTIVE_WIDTH", "1024"))
PANORAMA_PERSPECTIVE_HEIGHT = int(os.getenv("PANORAMA_PERSPECTIVE_HEIGHT", "1024"))
PANORAMA_PERSPECTIVE_FOV = float(os.getenv("PANORAMA_PERSPECTIVE_FOV", "90"))
PANORAMA_HORIZONTAL_YAWS = tuple(
    float(item.strip())
    for item in os.getenv("PANORAMA_HORIZONTAL_YAWS", "0,90,180,270").split(",")
    if item.strip()
)
# 迭代中：向下观察透视图的投影实现保留，但当前 Task 2/3 链路暂未调用。
PANORAMA_INCLUDE_DOWNWARD = False
PANORAMA_DOWNWARD_PITCH = float(os.getenv("PANORAMA_DOWNWARD_PITCH", "-20"))
PANORAMA_ASPECT_TOLERANCE = float(os.getenv("PANORAMA_ASPECT_TOLERANCE", "0.01"))
PANORAMA_STRICT_ASPECT = os.getenv(
    "PANORAMA_STRICT_ASPECT", "true"
).strip().lower() in {"1", "true", "yes"}

SCENE_UNDERSTANDING_ENABLED = os.getenv(
    "SCENE_UNDERSTANDING_ENABLED", "false"
).strip().lower() in {"1", "true", "yes"}

# RAG 默认关闭。开启后可选择本地 TF-IDF 或 Qwen 远程 Embedding 检索。
RAG_ENABLED = os.getenv("RAG_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
RAG_TOP_K = max(1, int(os.getenv("RAG_TOP_K", "4")))
RAG_MIN_SCORE = max(0.0, float(os.getenv("RAG_MIN_SCORE", "0.05")))
RAG_INCLUDE_REPOSITORY_SOURCES = os.getenv(
    "RAG_INCLUDE_REPOSITORY_SOURCES", "true"
).strip().lower() in {"1", "true", "yes"}
RAG_RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "auto").strip().lower()
if RAG_RETRIEVAL_MODE not in {"auto", "tfidf", "qwen_embedding"}:
    raise ValueError(
        "RAG_RETRIEVAL_MODE 仅支持 auto、tfidf 或 qwen_embedding"
    )
RAG_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "text-embedding-v4"
).strip()
RAG_EMBEDDING_DIMENSIONS = max(
    64, int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1024"))
)
RAG_EMBEDDING_BATCH_SIZE = max(
    1, int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10"))
)
RAG_EMBEDDING_TIMEOUT = max(
    1.0, float(os.getenv("RAG_EMBEDDING_TIMEOUT", "30"))
)
RAG_EMBEDDING_FALLBACK_TO_TFIDF = os.getenv(
    "RAG_EMBEDDING_FALLBACK_TO_TFIDF", "true"
).strip().lower() in {"1", "true", "yes"}

# DeepSeek 仅用于离线知识整理；不会受 RUN_MODE 自动触发，CLI 必须显式 --execute。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
).rstrip("/")
DEEPSEEK_FLASH_MODEL = os.getenv(
    "DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"
).strip()
DEEPSEEK_PRO_MODEL = os.getenv(
    "DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"
).strip()
DEEPSEEK_KNOWLEDGE_MAX_TOKENS = max(
    4000, int(os.getenv("DEEPSEEK_KNOWLEDGE_MAX_TOKENS", "16000"))
)
DEEPSEEK_KNOWLEDGE_THINKING = os.getenv(
    "DEEPSEEK_KNOWLEDGE_THINKING", "false"
).strip().lower() in {"1", "true", "yes"}

# ---------- API / 运行模式 ----------
_generic_llm_key = os.getenv("LLM_API_KEY", "").strip()
_qwen_api_key = os.getenv("QWEN_API_KEY", "").strip()

if _generic_llm_key:
    LLM_API_KEY = _generic_llm_key
    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-plus").strip()
elif _qwen_api_key:
    # 兼容已有的阿里云百炼变量，不要求用户复制或改名 API Key。
    LLM_API_KEY = _qwen_api_key
    LLM_BASE_URL = os.getenv(
        "QWEN_OPENAI_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    LLM_MODEL = os.getenv("QWEN_MODEL", "qwen3.7-plus").strip()
else:
    LLM_API_KEY = ""
    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-plus").strip()

# RAG Embedding 默认复用 Qwen/LLM 的兼容接口和密钥，也可单独配置。
# 不在日志、会话或缓存中保存 Key。
RAG_EMBEDDING_API_KEY = os.getenv(
    "RAG_EMBEDDING_API_KEY", LLM_API_KEY
).strip()
RAG_EMBEDDING_BASE_URL = os.getenv(
    "RAG_EMBEDDING_BASE_URL", LLM_BASE_URL
).strip().rstrip("/")

WORLDLABS_API_KEY = os.getenv("WORLDLABS_API_KEY", "").strip()
WORLDLABS_BASE_URL = os.getenv("WORLDLABS_BASE_URL", "https://api.worldlabs.ai").rstrip("/")
# Marble 模型：marble-1.1（默认）/ marble-1.1-plus（更大户外场景，耗更多 credits）
WORLDLABS_MODEL = os.getenv("WORLDLABS_MODEL", "marble-1.1").strip()

RUN_MODE = os.getenv("RUN_MODE", "auto").lower()  # auto | mock | live


def use_mock_llm() -> bool:
    if RUN_MODE == "mock":
        return True
    if RUN_MODE == "live":
        return False
    return not bool(LLM_API_KEY)


def use_mock_worldlabs() -> bool:
    if RUN_MODE == "mock":
        return True
    if RUN_MODE == "live":
        return False
    return not bool(WORLDLABS_API_KEY)


# ---------- 豆包 Seedream（火山方舟图生图）----------
SEEDREAM_API_KEY = os.getenv("SEEDREAM_API_KEY", os.getenv("ARK_API_KEY", "")).strip()
SEEDREAM_BASE_URL = os.getenv(
    "SEEDREAM_BASE_URL", "https://ark.cn-beijing.volces.com"
).rstrip("/")
SEEDREAM_MODEL = os.getenv("SEEDREAM_MODEL", "doubao-seedream-5-0-260128").strip()
# 2K / 3K / 4K / 2048x1024 / auto（auto=按原图比例估算）
SEEDREAM_SIZE = os.getenv("SEEDREAM_SIZE", "2K").strip()
SEEDREAM_RESPONSE_FORMAT = os.getenv("SEEDREAM_RESPONSE_FORMAT", "url").strip()
SEEDREAM_WATERMARK = os.getenv("SEEDREAM_WATERMARK", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)


def use_mock_seedream() -> bool:
    if RUN_MODE == "mock":
        return True
    if RUN_MODE == "live":
        return False
    return not bool(SEEDREAM_API_KEY)



EXPERIENCE_KEYS = [
    "comfort",                    # 舒适度
    "naturalness",                # 自然感
    "safety",                     # 安全感
    "relaxation",                 # 放松感
    "environmental_disturbance",  # 环境干扰感（反向指标）
    "stay_intention",             # 可停留意愿
    "overall_impression",         # 总体感
]

EXPERIENCE_LABELS_ZH = {
    "comfort": "舒适度",
    "naturalness": "自然感",
    "safety": "安全感",
    "relaxation": "放松感",
    "environmental_disturbance": "环境干扰感",
    "stay_intention": "可停留意愿",
    "overall_impression": "总体感",
}

# 所有指标均采用 1~5 分。除环境干扰感外，分值越高表示体验越好。
EXPERIENCE_SCALE = {"min": 1.0, "max": 5.0, "neutral": 3.0}
EXPERIENCE_DIRECTIONS = {
    "comfort": "higher_is_better",
    "naturalness": "higher_is_better",
    "safety": "higher_is_better",
    "relaxation": "higher_is_better",
    "environmental_disturbance": "lower_is_better",
    "stay_intention": "higher_is_better",
    "overall_impression": "higher_is_better",
}

# 兼容 v2 的五指标输入。新流程输出始终使用 EXPERIENCE_KEYS 中的七个键。
LEGACY_EXPERIENCE_ALIASES = {
    "restoration": "relaxation",
    "stay": "stay_intention",
    "pleasure": "overall_impression",
}


def normalize_experience_values(values: dict | None) -> dict[str, float]:
    """把新旧体验字段统一为七项 1~5 分浮点值。"""
    raw = dict(values or {})
    for old_key, new_key in LEGACY_EXPERIENCE_ALIASES.items():
        if new_key not in raw and old_key in raw:
            raw[new_key] = raw[old_key]

    neutral = EXPERIENCE_SCALE["neutral"]
    lower = EXPERIENCE_SCALE["min"]
    upper = EXPERIENCE_SCALE["max"]
    return {
        key: min(upper, max(lower, float(raw.get(key, neutral))))
        for key in EXPERIENCE_KEYS
    }

# ---------- 七项形态要素（图像解析 + 制图员输出）----------
MORPH_KEYS = [
    "green_view",          # 绿视率
    "blue_view",           # 蓝视率
    "sky_view",            # 天空可视率
    "built_ratio",         # 人造物占比
    "color_richness",      # 有效色彩数量
    "edge_density",        # 边缘密度
    "skyline_variance",    # 天际线变化率
]

MORPH_LABELS_ZH = {
    "green_view": "绿视率",
    "blue_view": "蓝视率",
    "sky_view": "天空可视率",
    "built_ratio": "人造物占比",
    "color_richness": "色彩丰富度",
    "edge_density": "边缘密度",
    "skyline_variance": "天际线变化率",
}

# 七项形态指标按计算公式定义的理论取值空间。
# 这些是输入校验和结果裁剪使用的物理边界，不是经验性“最佳区间”。
# 绿视率等比例指标统一使用 0~1；色彩丰富度使用有效颜色数 0~24。
MORPH_BOUNDS = {
    "green_view": (0.0, 1.0),
    "blue_view": (0.0, 1.0),
    "sky_view": (0.0, 1.0),
    "built_ratio": (0.0, 1.0),
    "edge_density": (0.0, 1.0),
    "color_richness": (0.0, 24.0),
    "skyline_variance": (0.0, 1.0),
}

# 语义分割后端：auto | gluoncv | segformer | fallback
# auto 优先 GluonCV（文章方案），不可用时回退 SegFormer，再回退 OpenCV
MORPH_SEG_BACKEND = os.getenv("MORPH_SEG_BACKEND", "auto").lower()

# GluonCV DeepLabV3（腾讯云文章方案，需 Python 3.8/3.9 + mxnet）
# 可选: deeplab_resnet101_ade | deeplab_resnet101_citys
GLUONCV_MODEL = os.getenv("GLUONCV_MODEL", "deeplab_resnet101_ade")

# SegFormer（现代 Windows / Python 3.10+ 推荐）
SEGFORMER_MODEL = os.getenv("SEGFORMER_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512")
