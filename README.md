"""
高密度情境下微空间优化 · 多智能体全流程工程 v2
================================================

## v2 流程变更

- **翻译官**：体验滑块原值→目标值 → 形态要素原值+目标值（不再输出设计意图文本）
- **制图员**：形态要素目标 → 自然语言修改方案（承接原提示词专家职责）
- **提示词专家**：已废弃，合并至制图员
- **学习 Agent**：占位接口，记录体验→形态翻译准确度（默认不启用修正）
- **多人体验**：支持修改前/修改后多人体验指标 JSON 输入
- **情景要素**：与全景图配套输入

## 目录结构

```
code/
  morph_metrics_extractor.py   # 工具：SegFormer 形态要素解析
  agents/
    translator_agent.py        # 翻译官：体验变化 → 形态目标
    cartographer_agent.py      # 制图员：形态目标 → 自然语言方案
    learning_agent.py          # 学习 Agent（占位）
    worldlabs_agent.py         # 工具：World Labs Pano Edit 文生图
    quality_checker_agent.py   # 质检员
    memory_agent.py            # 记忆入库
    prompt_expert_agent.py     # @deprecated
  pipeline/orchestrator.py     # 全流程编排
  app/gradio_app.py            # Gradio 前端
  app/api_server.py            # FastAPI 接口
```

## 快速开始

```bash
cd code
python -m pip install -r requirements.txt
copy .env.example .env

# 单独测试形态解析
python morph_metrics_extractor.py ..\JPG素材\某全景.jpg --fallback

# 启动前端
python app/gradio_app.py

# 命令行一键演示
python run_demo.py path/to/pano.jpg
```

## 数据流

1. 上传全景 + 情景要素 +（可选）修改前多人体验
2. `morph_metrics_extractor` 解析 7 维形态基线
3. 前端调节五个体验滑块 → **翻译官** → 形态原值/目标值 → 人工干预①
4. **制图员** 生成自然语言修改方案 → 人工干预②
5. **World Labs** 文生图 → 质检
6. 填写修改后多人体验 → **记忆 Agent** + **学习 Agent** 入库

每个 Agent 文件头部注释写清了：输入 / 输出 / 输出到哪里 / 调用方式。

## 各代码文件职责说明

### 根目录

| 文件 | 职责 |
|------|------|
| `config.py` | 全局配置：路径、体验/形态维度定义、API Key、SegFormer 模型名、MOCK 模式判断 |
| `morph_metrics_extractor.py` | **独立工具**：用 SegFormer（或 OpenCV fallback）解析全景图，输出 7 个形态要素指标 |
| `run_demo.py` | 命令行一键演示入口，跳过人工干预跑完全流程 |
| `requirements.txt` | Python 依赖清单 |
| `.env.example` | 环境变量模板（`LLM_API_KEY`、`WORLDLABS_API_KEY` 等） |

### agents/ — 智能体

| 文件 | 职责 |
|------|------|
| `translator_agent.py` | **翻译官**：接收体验滑块原值→目标值，结合映射规则与知识库，输出形态要素原值 + 目标值 |
| `cartographer_agent.py` | **制图员**：接收确认后的形态目标，生成可被文生图模型理解的自然语言修改方案 |
| `learning_agent.py` | **学习 Agent（占位）**：记录体验→形态翻译是否准确，预留多轮学习修正接口（默认不启用） |
| `worldlabs_agent.py` | **文生图工具**：按图片完整文件名从 `assets/` 取图，调 Marble API，结果写入 `TargetIMG/` |
| `quality_checker_agent.py` | **质检员**：对修改后全景重新解析形态要素，与目标对比输出偏差报告 |
| `memory_agent.py` | **记忆 Agent**：将全流程关键数据（体验、形态、方案、多人体验）写入本地知识库 |
| `prompt_expert_agent.py` | `@deprecated` 已废弃，逻辑合并至 `cartographer_agent.py`，仅保留向后兼容 |
| `__init__.py` | Agent 包统一导出 |

### pipeline/ — 流程编排

| 文件 | 职责 |
|------|------|
| `orchestrator.py` | **全流程编排器**：串联解析→翻译官→制图员→文生图→质检→记忆/学习，管理 session 状态与持久化 |
| `__init__.py` | pipeline 包标识 |

### app/ — 前端与接口

| 文件 | 职责 |
|------|------|
| `gradio_app.py` | **Gradio 人机协同界面**：上传全景、情景要素、体验滑块、形态干预、方案润色、结果展示、知识库入库 |
| `api_server.py` | **FastAPI REST 接口**：与 Gradio 等价的 HTTP 端点，供组员不经前端直接联调 |

### schemas/ — 数据模型

| 文件 | 职责 |
|------|------|
| `models.py` | 全流程共用的 Pydantic 模型：体验指标、形态指标、翻译结果、修改方案、质检报告、记忆条目等 |
| `__init__.py` | schemas 包统一导出 |

### utils/ — 工具函数

| 文件 | 职责 |
|------|------|
| `llm_client.py` | OpenAI 兼容 LLM 客户端；无 API Key 时返回 None，各 Agent 走规则兜底 |
| `__init__.py` | utils 包标识 |

### knowledge_base/ — 本地知识库

| 文件 | 职责 |
|------|------|
| `kb_store.py` | 知识库读写与 Few-shot 检索（体验旋钮余弦相似度匹配历史案例） |
| `data/mapping_rules.json` | 体验感受→形态要素的默认映射规则（翻译官使用） |
| `data/memories.json` | 历史记忆条目（各 Agent 决策时检索参考） |
| `data/learning_feedback.json` | 学习 Agent 反馈记录（翻译准确度，默认 `enabled: false`） |
| `__init__.py` | knowledge_base 包标识 |

### outputs/ — 运行时输出（自动生成）

| 目录/文件 | 职责 |
|-----------|------|
| `outputs/images/` | World Labs / MOCK 生成的修改后全景图 |
| `outputs/sessions/` | 每次运行的 session 状态 JSON（断点续跑、调试） |
| `uploads/` | 用户上传的全景图缓存 |

### _extracted_text/ — 项目文档文本（参考）

| 文件 | 职责 |
|------|------|
| `第二日技术方案v2.txt` | 核心技术方案：旋钮→Agent→HITL→World Labs→知识闭环 |
| `metrics_extract_route.txt` | 数据指标提取技术路线（SegFormer 7 可行指标） |
| `metrics_definition.txt` | 形态/体验指标定义与计算公式 |
| 其余 `.txt` | 工作营任务书、科学问题阐述、完整 Pipeline 等背景文档 |
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
