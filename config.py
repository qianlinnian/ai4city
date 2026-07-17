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
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

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


# ---------- 体验感受旋钮（Step1 / 翻译官输入）----------
EXPERIENCE_KEYS = [
    "comfort",       # 舒适度
    "restoration",   # 恢复感 / 放松感
    "safety",        # 安全感
    "pleasure",      # 愉悦感
    "stay",          # 可停留意愿
]

EXPERIENCE_LABELS_ZH = {
    "comfort": "舒适度",
    "restoration": "恢复感",
    "safety": "安全感",
    "pleasure": "愉悦感",
    "stay": "可停留意愿",
}

# ---------- 形态要素（图像解析 + 制图员输出）----------
MORPH_KEYS = [
    "green_view",          # 绿视率
    "blue_view",           # 蓝视率
    "sky_view",            # 天空可视率
    "built_ratio",         # 人造物占比
    "edge_density",        # 边缘密度
    "color_richness",      # 有效色彩数量
    "skyline_variance",    # 天际线变化率
]

MORPH_LABELS_ZH = {
    "green_view": "绿视率",
    "blue_view": "蓝视率",
    "sky_view": "天空可视率",
    "built_ratio": "人造物占比",
    "edge_density": "边缘密度",
    "color_richness": "有效色彩数量",
    "skyline_variance": "天际线变化率",
}

# 形态要素合理区间（用于制图员约束 / 质检）
MORPH_BOUNDS = {
    "green_view": (0.05, 0.55),
    "blue_view": (0.0, 0.25),
    "sky_view": (0.05, 0.55),
    "built_ratio": (0.15, 0.80),
    "edge_density": (0.02, 0.25),
    "color_richness": (1.0, 12.0),
    "skyline_variance": (0.005, 0.15),
}

# SegFormer 模型（形态要素解析）
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade20k"
