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

for _p in (KB_DIR, UPLOAD_DIR, IMAGE_OUT_DIR, SESSION_DIR, ASSETS_DIR, TARGET_IMG_DIR):
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


# ---------- 七项 VR 体验感受（Task 2 / 翻译官输入）----------
# 指标名称以《指标定义及计算方式(1).xlsx》中的“VR体验感受”工作表为准。
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

# SegFormer 模型（形态要素解析）
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade20k"
