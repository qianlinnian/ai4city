"""
高密度情境下微空间优化 · 多智能体全流程工程 v2
================================================

## v2 流程变更

- **翻译官**：同一图像全部参与者逐人七项评分 + 体验目标 + 形态初始值 + 原始全景 → 七项形态目标（LangChain 多模态 Prompt；规则仅离线兜底）
- **制图员**：原始全景 + 确认后的形态目标 + 专家建议 → 结构化空间布局方案 + World Labs 修改文本
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
    cartographer_agent.py      # 制图员：形态目标 → 结构化空间布局方案与修改文本
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
3. 前端调节七个体验滑块 → **翻译官** → 形态原值/目标值 → 人工干预①
4. **制图员** 生成结构化空间布局方案与自然语言修改文本 → 人工干预②
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
| `translator_agent.py` | **翻译官**：逐人保留全部七项评分，以 LangChain 多模态 Prompt 直接生成七项形态目标；RAG 暂留接口，规则仅作无模型兜底 |
| `cartographer_agent.py` | **制图员**：接收原图、确认后的形态目标和专家建议，输出对象级空间布局方案及可执行修改文本 |
| `learning_agent.py` | **学习 Agent（占位）**：记录体验→形态翻译是否准确，预留多轮学习修正接口（默认不启用） |
| `worldlabs_agent.py` | **文生图工具封装**：调用 World Labs Pano Edit API，无 Key 时 MOCK 生成演示图 |
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
| `llm_client.py` | LangChain 模型适配层：`ChatPromptTemplate | ChatOpenAI | StrOutputParser`；支持全景图输入，无 API Key 时返回 None，各 Agent 走本地规则兜底 |
| `__init__.py` | utils 包标识 |

### knowledge_base/ — 本地知识库

| 文件 | 职责 |
|------|------|
| `kb_store.py` | 旧版知识库读写与案例检索能力；当前 Task 2/3 默认不调用 |
| `data/mapping_rules.json` | 七项体验→七项形态的基础映射参数（启发式初值，待实验校准） |
| `data/experience_morph_cases.json` | 旧版非实测案例样例；当前不参与 Task 2/3 计算 |
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
