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

# ---------- 路径 ----------
KB_DIR = ROOT / "knowledge_base" / "data"
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_OUT_DIR = OUTPUT_DIR / "images"
SESSION_DIR = OUTPUT_DIR / "sessions"
# 文生图工具：按图片完整文件名从 assets/ 取原图，结果写入 TargetIMG/
ASSETS_DIR = ROOT / "assets"
TARGET_IMG_DIR = ROOT / "TargetIMG"
# 前端只读取后端维护的数据目录，不要求浏览器重复上传原图或项目大表。
DATA_DIR = Path(os.getenv("AI4CITY_DATA_DIR", str(ROOT / "data"))).resolve()
PANORAMA_DIR = Path(os.getenv("AI4CITY_PANORAMA_DIR", str(DATA_DIR))).resolve()
METRICS_TABLE_DIR = Path(
    os.getenv("AI4CITY_METRICS_TABLE_DIR", str(DATA_DIR))
).resolve()
SCENE_MANIFEST_PATH = Path(
    os.getenv("AI4CITY_SCENE_MANIFEST", str(DATA_DIR / "scenes.csv"))
).resolve()
_metrics_table_path_raw = os.getenv("AI4CITY_METRICS_TABLE", "").strip()
METRICS_TABLE_PATH = (
    Path(_metrics_table_path_raw).resolve() if _metrics_table_path_raw else None
)

for _p in (
    KB_DIR,
    UPLOAD_DIR,
    IMAGE_OUT_DIR,
    SESSION_DIR,
    ASSETS_DIR,
    TARGET_IMG_DIR,
    DATA_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

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
