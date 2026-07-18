"""
高密度情境下微空间优化 · 多智能体全流程工程 v2
================================================

## 本机快速启动（Windows）

### 1. 准备环境

```powershell
cd D:\GitHubZY\ai4city
python -m pip install -r requirements.txt
```

`.env` 已按本机路径配置（`DATA_DIR=D:\GitHubZY\ai4city`），API Key 直接写在 `.env` 中即可。

### 2. 放入原图（文生图必需）

把全景原图放到：

```text
D:\GitHubZY\ai4city\assets\
```

文件名需能与 `filled_metrics.xlsx` 的 **B 列** 用**前 26 位**对齐，例如：

- Excel B 列：`VID_20260709_164915_00_114`
- 原图可为：`VID_20260709_164915_00_114_2026-07-16_14-22-01.jpg`

分析图已在仓库内，按同名前缀自动匹配：

| 类型 | 目录 | 文件名前缀 |
|------|------|------------|
| 边缘密度 | `edge_density_maps/` | `edge_` |
| 语义分割 | `segmentation_results/` | `seg_` |
| 天际线 | `skyline_boundary_maps/` | `skyline_` |

指标表：`filled_metrics.xlsx`（根目录）

> 若 `assets/` 暂时为空，前端下拉会回退为边缘图 stem，仍可加载 Excel 指标与三张分析图；**Seedream 文生图**前必须补上对应原图。

### 3. 启动前端

```powershell
cd D:\GitHubZY\ai4city
python app/gradio_app.py
```

浏览器打开：http://127.0.0.1:7860

### 4.（可选）启动 FastAPI

```powershell
cd D:\GitHubZY\ai4city
uvicorn app.api_server:app --reload --port 8000
```

常用接口：

- `GET  /scenes` — 可选场景列表
- `GET  /scenes/{image_name}` — Excel 指标 + 分析图路径
- `POST /pipeline/start_from_dataset` — 从数据集启动 session
- `POST /pipeline/run_translator` → `confirm_morph` → `confirm_plan` → `generate`

---

## 前端操作流程

1. **下拉选图** → 自动展示原图 / edge / seg / skyline，并填入情景要素、形态基线、九人体感  
2. **① 确认加载场景** → 写入 session（形态基线来自 Excel，不重跑分割）  
3. **调节体验目标滑块** → **② 翻译官** → 得到目标形态要素  
4. **可改形态滑块** → **③ 制图员** → 自然语言修改方案  
5. **润色方案** → **④ Seedream 文生图** → 结果从 `TargetIMG/` 读回前端并质检  
6. **⑤ 填写修改后体验** → 写入本地知识库  

---

## Excel 列映射

| 列 | 含义 |
|----|------|
| B | 图片名称（前 26 位唯一匹配） |
| C–I | 情景要素（观测时间/天气/人流/空间类型/声音/维护/交通） |
| J–P | 形态要素（绿视率、蓝视率、人造物占比、天空可视率、色彩丰富度、边缘密度、天际线变化率） |
| Q 起每 7 列 | 一人体感（共 9 人）：舒适度、自然感、安全感、放松感、环境干扰感、可停留意愿、总体感 |

---

## 目录结构

```
ai4city/
  assets/                      # 全景原图（需自行放入）
  TargetIMG/                   # Seedream 生成结果
  edge_density_maps/           # edge_* 分析图
  segmentation_results/        # seg_* 分析图
  skyline_boundary_maps/       # skyline_* 分析图
  filled_metrics.xlsx          # 情景 + 形态 + 九人体感
  config.py                    # 路径与模型配置
  agents/                      # 翻译官 / 制图员 / Seedream / 质检 / 记忆 …
  pipeline/orchestrator.py     # 全流程编排
  app/gradio_app.py            # Gradio 前端
  app/api_server.py            # FastAPI
  utils/scene_data.py          # Excel + 分析图加载
```

## 数据流

1. 选图 → `utils/scene_data.py` 读 Excel + 三张分析图  
2. 体验滑块 → **翻译官** → 形态目标 → 人工干预①  
3. **制图员** → 自然语言方案 → 人工干预②  
4. **Seedream** 图生图 → `TargetIMG/` → 质检  
5. 修改后体验 → **记忆 Agent** + **学习 Agent**

各 Agent 文件头部注释写清了输入 / 输出；编排层尽量不改 Agent 接口，仅切换文生图为 Seedream。

## 环境变量要点（见 `.env.example`）

| 变量 | 说明 |
|------|------|
| `DATA_DIR` | 数据根目录，本机为 `D:\GitHubZY\ai4city` |
| `LOAD_DATA_DIR` | 原图目录（相对 `DATA_DIR` 或绝对路径），默认 `assets` |
| `QWEN_API_KEY` / `LLM_*` | 翻译官 / 制图员多模态 LLM |
| `SEEDREAM_API_KEY` | 豆包 Seedream 图生图 |
| `RUN_MODE` | `auto` / `mock` / `live` |
"""
