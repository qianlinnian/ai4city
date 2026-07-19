"""
高密度情境下微空间优化 · 多智能体全流程工程 v2
================================================

## v2 流程变更

- **场景理解**：原始 2:1 全景 → 1 张概览 + 4 张带方位的球面透视图 → Qwen 结构化场景清单
- **翻译官**：同一图像全部参与者逐人七项评分 + 体验目标 + 形态初始值 + 场景清单 → 七项形态目标
- **制图员**：修改前后形态指标 + 场景清单 + 专家建议 → 结构化空间布局方案与最终编辑文本
- **可插拔 RAG**：默认关闭；开启后以 `ai4city-data/knowledge` 中的规范 PDF 为主要知识源，支持 Qwen `text-embedding-v4` 语义检索与本地 TF-IDF 回退，仓库规则仅作补充
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

# 启动前端
$env:AI4CITY_DATA_DIR="D:\course\ai4city-data"
$env:RUN_MODE="mock"
python app/gradio_app.py

# 命令行一键演示
python run_demo.py path/to/pano.jpg
```

## 数据流

1. 前端从只读后端目录选择全景图和项目大表行，不重复上传原始数据。
2. Task 1 已离线完成；在线流程直接读取大表中的七项形态基线。
3. 全景模块生成概览和四向透视图；Qwen 先输出结构化场景清单。
4. 前端调节七项体验目标 → **翻译官** → 七项形态目标 → 专家确认。
5. **制图员** 使用场景清单、形态前后值、RAG 参考和专家建议生成同一份可编辑布局/提示词对象。
6. Task 4 图像生成不属于本次 Task 2/3 自动测试范围。

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
| `translator_agent.py` | **翻译官**：逐人保留全部七项评分，以 LangChain Prompt 生成七项形态目标；RAG 开启时注入检索参考，规则仅作无模型兜底 |
| `cartographer_agent.py` | **制图员**：接收原图、确认后的形态目标和专家建议，输出对象级空间布局方案及可执行修改文本 |
| `scene_understanding_agent.py` | **场景理解**：调用现有 LangChain/Qwen 多图链，输出带证据视图的结构化场景清单，失败时显式降级 |
| `reasonableness.py` | **合理性检查**：范围、变化幅度、指标冲突、水体和固定区域等内部警告，不静默改写目标 |
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
| `llm_client.py` | LangChain 模型适配层；支持单图与带 yaw/pitch/FOV 说明的多图消息，无 Key 时返回 None |
| `panorama_views.py` | 正确的等距柱状→透视球面投影、左右接缝环绕、确定性缓存和视图元数据 |
| `__init__.py` | utils 包标识 |

### knowledge_base/ — 本地知识库

| 文件 | 职责 |
|------|------|
| `rag_provider.py` | `NullRagProvider`、Qwen Embedding 与本地字符 TF-IDF Provider；只读提取 knowledge PDF、保留页码并缓存到项目输出目录 |
| `knowledge_curator.py` | DeepSeek 离线知识整理：Flash 批处理、可选 Pro 复核、原文证据校验和结构化草稿 |
| `data/mapping_rules.json` | 七项体验→七项形态的基础映射参数（启发式初值，待实验校准） |
| `data/experience_morph_cases.json` | 旧版非实测案例样例；当前不参与 Task 2/3 计算 |
| `data/spatial_rules.json` | Task 3 空间约束参考；围合度、通透度、边界层数不属于七项形态指标 |
| `data/memories.json` | 历史记忆条目（各 Agent 决策时检索参考） |
| `data/learning_feedback.json` | 学习 Agent 反馈记录（翻译准确度，默认 `enabled: false`） |
| `__init__.py` | knowledge_base 包标识 |

### outputs/ — 运行时输出（自动生成）

| 目录/文件 | 职责 |
|-----------|------|
| `outputs/images/` | World Labs / MOCK 生成的修改后全景图 |
| `outputs/sessions/` | 每次运行的 session 状态 JSON（断点续跑、调试） |
| `outputs/panorama_views/` | Task 2/3 场景理解派生图及 `views.json`；相同原图与配置可直接复用 |
| `outputs/knowledge_drafts/` | DeepSeek 生成的知识草稿；仅 `program_validated` 记录可由发布脚本写入正式 RAG，其他状态继续暂缓 |
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
