# Task 2–3 基础框架

本分支实现“七项体感目标 → 七项形态目标 → 空间布局方案与修改文本”的可运行骨架，便于后续接入真实 VR 数据、专家规则和正式 LLM。

## 参数总表

- 元数据：场景编号、图片名称。
- 七项情景要素：观测时间、观测天气、人流量、空间类型、声音类型、管理维护状态、交通流量。
- 七项形态要素：绿视率、蓝视率、天空可视率、人造物占比、色彩丰富度、边缘密度、天际线变化率。
- 七项体感指标：舒适度、自然感、安全感、放松感、环境干扰感、可停留意愿、总体感。

## 七项体感指标

指标名称来自仓库根目录 `指标定义及计算方式(1).xlsx` 的“VR体验感受”工作表。

| API 键 | 中文名称 | 量表 | 方向 |
|---|---|---:|---|
| `comfort` | 舒适度 | 1–5 | 越高越好 |
| `naturalness` | 自然感 | 1–5 | 越高越好 |
| `safety` | 安全感 | 1–5 | 越高越好 |
| `relaxation` | 放松感 | 1–5 | 越高越好 |
| `environmental_disturbance` | 环境干扰感 | 1–5 | 越低越好 |
| `stay_intention` | 可停留意愿 | 1–5 | 越高越好 |
| `overall_impression` | 总体感 | 1–5 | 越高越好 |

旧字段 `restoration`、`pleasure`、`stay` 会分别兼容映射为 `relaxation`、`overall_impression`、`stay_intention`；新接口和输出只使用七项新字段。

## Task 2：翻译官 Agent

### 七项形态指标的公式取值空间

硬校验只采用指标计算公式的理论范围，不把经验性效用区间作为上下限：

| 指标 | 公式取值空间 | 程序表示 |
|---|---:|---|
| 绿视率 | 0%～100% | `0.0～1.0` |
| 蓝视率 | 0%～100% | `0.0～1.0` |
| 天空可视率 | 0%～100% | `0.0～1.0` |
| 人造物占比（像素占比） | 0%～100% | `0.0～1.0` |
| 色彩丰富度（有效颜色数） | 0～24 | `0.0～24.0` |
| 边缘密度 | 0～1 | `0.0～1.0` |
| 天际线变化率 | 0%～100% | `0.0～1.0` |

绿视率的有效增长区、平台区以及基于 P25/P50/P75 的分档仅作为 Prompt 的软参考，不能截断合法输入或目标结果。

输入：

- 原始 JPG（配置 LLM 后可作为多模态输入）
- 同一图像全部参与者的逐人七项体感评分（每项 1～5，完整输入，不求平均）
- 一组七项体感目标值
- Task 1 输出的七项形态要素原始值
- 情景要素

基础流程：

1. 严格校验每位参与者七项评分；缺项或出现 1～5 之外的值（例如误填 6）立即报错，不自动截断。
2. 有 LLM Key 时，由 LangChain 把全部逐人评分、体感目标、七项形态初始值、情景与原图交给多模态模型。
3. 模型直接输出且只输出七项形态目标 JSON，不先运行规则或案例融合。
4. RAG 默认关闭；开启时以本地字符 TF-IDF 检索项目规则、指标定义、专家规则和案例，检索内容仅作不可信参考。
5. 无 LLM、调用失败或模型输出越界时，才对每位参与者分别应用映射规则，再对形态目标取中位数兜底。
6. 程序计算形态前后差值，并保留专家编辑确认入口。

## Task 3：制图员 Agent

输入：

- 原始 JPG 经多视图模块生成的结构化场景理解结果
- 修改前后的七项形态要素值
- 七项体感目标、情景要素和专家建议
- 可选本地 RAG（默认关闭）

输出 `ModificationPlan`：

- `plan_summary`：空间布局方案摘要
- `object_actions`：新增、移除或调整的对象、位置、数量、属性和理由
- `spatial_relations`：对象之间及对象与通行空间的关系
- `unchanged_regions`：明确保持不变的区域
- `constraints`：建筑、道路、景观、全景接缝与真实性约束
- `draft_text`：专家可编辑、可直接传给 World Labs Pano Edit 的修改文本
- `rag_references`：使用的历史案例 ID

有 LLM 时通过 LangChain 调用模型并要求返回严格 JSON；无 LLM 时由形态差值规则生成完整的结构化方案和修改文本。

## LangChain 调用层

- 文本链：`ChatPromptTemplate | ChatOpenAI | StrOutputParser`。
- 多模态链：使用 LangChain `SystemMessage` / `HumanMessage`，发送一张 2:1 概览和四张 yaw=0/90/180/270 的水平透视图；每张图前附 yaw、pitch、FOV 和 view_id。向下观察视图仍在迭代，当前链路暂未调用。
- 场景清单：Qwen 先返回道路、建筑、入口、植被、水体、家具、基础设施、可编辑对象、固定区域、空间关系、接缝约束、歧义和证据视图。失败时返回空清单与 `degraded`，Task 2/3 继续兜底。
- RAG：`RAG_ENABLED=false` 时不建立索引、不检索；开启时本地 TF-IDF 返回 `text/source/chunk_id/score/metadata`。Prompt 明确其不是系统指令，不得新增第八项指标或覆盖专家确认。
- 兜底：未配置 Key、依赖不可用或调用失败时，使用本地映射参数与规则方案，便于离线联调。

## 基础参数与示例数据

- `knowledge_base/data/mapping_rules.json`：七项体感到七项形态的第一版启发式系数。
- `knowledge_base/data/experience_morph_cases.json`：根据 `data/p1.jpg`、`data/p2.jpg`、现有图像特征和历史启发式评分整理的两条种子案例。

种子案例不是专家实测值，也不表达因果关系；当前版本不把它们注入 Task 2/3。后续启用 RAG 时，应以真实 VR 前后测、课本知识和专家修正记录替换或校准。

## 全景图片与缺失数据补全

- 推荐输入已拼接完成的等距柱状投影全景 `JPG`，画面比例优先为 2:1，并尽量保留原始分辨率；`PNG` 也可读取。
- 相机原始 `INSP` 文件不直接交给本项目，应先在 Insta360 Studio 中导出为全景 JPG，再按表格中的“图片名称”与场景逐行对应。
- 绿视率、蓝视率、天空可视率、人造物占比由 SegFormer 语义分割估算；色彩丰富度、边缘密度、天际线变化率由 OpenCV 图像算法计算。
- 七项体感值不能被当作仅凭图片得到的真实 VR 实测值。联调模拟值必须明确标注为“模拟/估算”，不能与表格中的逐人真实体验数据混用。
- 建议补全数据额外保留 `data_nature`（`measured` / `computed` / `synthetic`）、`generation_method` 和 `confidence` 字段，保证数据来源可追溯。

无需 API Key 的演示：

```bash
python examples/task2_3_demo.py
```

## 全景视图配置与缓存

- 输入默认严格要求 2:1；非 2:1 由 `PANORAMA_STRICT_ASPECT` 决定拒绝或明确警告。
- 概览默认 `2048×1024`；水平透视图默认 `1024×1024`、FOV 90°，可配置为 `1536×1536`。
- 向下观察视图的球面投影代码和测试继续保留，但生产配置固定关闭，当前 Task 2/3 不生成或发送该视图；后续迭代可通过显式构造 `PanoramaViewConfig(include_downward=True)` 重新启用。
- 球面投影使用 NumPy/OpenCV，横向经度取模并以环绕边界插值处理左右接缝。
- 输出路径为 `outputs/panorama_views/<原图名_内容哈希_配置哈希>/`，同一原图和配置再次运行直接复用。
- 原始图和 `D:\course\ai4city-data` 始终只读；缓存目录若配置到数据目录内会立即拒绝。

## Task 2 内部合理性检查

翻译官生成七项目标后记录内部检查结果，但不增加或改写对外七项目标：理论范围、单次变化幅度、绿视率/天空可视率冲突、固定建筑和道路可实施性、天际线约束，以及无真实水体时蓝视率异常提升。检查只能给出警告或显式降级，最终取舍仍由专家确认。

## RAG 知识范围

主要知识源为只读目录 `D:\course\ai4city-data\knowledge` 中的规范 PDF。系统使用现有 `pdftotext` 按页提取，保留 PDF 文件名、页码与分块编号，并把派生文本缓存到项目 `outputs/rag_cache/`；不会修改原 PDF。仓库内 `knowledge_base/data`、本框架文档和指标定义只作为补充来源。PDF 和长文本按约 900 字切块，中文和英文以 2～5 字符 n-gram TF-IDF 检索；默认约 75% 返回名额优先给达到相关度门槛的外部 PDF 块。

知识整理器和本地 RAG 会发现知识源目录中的全部支持文件；不再维护按文件名排除的隐式名单。`spatial_rules.json` 中的围合度、界面通透度和边界层数仅是 Task 3 对象布局参考，不得进入七项形态目标。

### DeepSeek 离线知识整理

`scripts/build_curated_knowledge.py` 可将 PDF 原文整理为带章节、条款、对象、约束、七指标关联和原文证据的结构化草稿。默认仅 dry-run；只有显式增加 `--execute` 才会调用 DeepSeek。Flash 负责批量整理，`--auto-pro` 只对证据校验失败、低置信度、模型主动标记复核或含高风险规范措辞的批次调用 Pro。结构化抽取默认显式关闭 V4 思考模式，并使用 16000 输出 tokens，避免推理内容挤占 JSON 输出额度；可分别通过 `DEEPSEEK_KNOWLEDGE_THINKING` 和 `DEEPSEEK_KNOWLEDGE_MAX_TOKENS` 调整。

中断后重复执行会跳过已有草稿。`--retry-invalid` 只重做已有但含校验错误的批次，`--retry-empty` 只重做状态为 `empty` 的批次；两者可同时使用，不需要用 `--force` 覆盖整份文档。

草稿写入 `outputs/knowledge_drafts`，不会修改 `ai4city-data`。每条记录必须保存 PDF 来源、页码和可在 OCR 原文中定位的连续引文；程序校验通过只表示格式和证据可定位，不等于专家确认。草稿经过人工抽检后才能进入正式 RAG 来源。

统一证据 Schema 使用 `page + line_ids + quote`。输入给 DeepSeek 的每一行都带稳定编号（如 `P0016-L0021`）；模型可跳过重复 OCR 行并组合多行，程序验证行号确实存在、页码一致且引文由对应行支持。证据校验优先使用去空白后的严格匹配；对 OCR 重复行，仅在证据不少于 12 个字符、字符 n-gram 覆盖率不低于 0.85 且存在足够长的连续锚点时接受 `fuzzy_ocr`，并在 JSON 中记录行号来源、匹配类型和分数。旧草稿会保守推断行号，无法可靠映射的记录继续保留 `needs_review`，不会伪造引用。已有草稿可在不产生 API 费用的情况下迁移和重新校验：

```powershell
python scripts/build_curated_knowledge.py --revalidate-existing
```

```powershell
# 只查看文档和批次数量，不调用 API
python scripts/build_curated_knowledge.py

# 先从正文页付费试跑一个批次，避免把封面和目录作为首个样本
python scripts/build_curated_knowledge.py --execute --start-page 10 --limit-batches 1

# 全量运行；需要复核的批次自动升级到 Pro
python scripts/build_curated_knowledge.py --execute --auto-pro
```

批量 API 调用属于网络 I/O，脚本使用线程并发而不是 CPU 多进程。默认 `--workers 1`，允许范围为 1～64；大批量任务建议先使用 16～32，根据账号并发配额、429 和网络稳定性调整。每个工作线程拥有独立 API 客户端，每个批次写独立 JSON，已完成批次可断点续跑。不要同时启动多个完全相同的命令，否则它们可能在启动时都判断同一批次尚未完成并重复计费。

```powershell
# 单进程内同时发出 4 个互相独立的请求
python scripts/build_curated_knowledge.py --include 公园设计规范 --start-page 10 --execute --workers 4

# 或在 4 个终端中分别运行，index 依次为 0、1、2、3
python scripts/build_curated_knowledge.py --execute --shard-count 4 --shard-index 0
```

### 正式 RAG 知识发布

`outputs/knowledge_drafts` 始终是草稿区；正式知识发布到 `rag_knowledge/published`，RAG 只读取后者。项目不设置额外专家批准层：记录经过统一程序门禁并得到 `program_validated` 后即可发布；`needs_review`、`empty`、缺少 OCR 行号或证据无法定位的记录不会进入正式知识库。

```powershell
# 查看可发布数量，不写文件
python scripts/publish_rag_knowledge.py --include 公园设计规范

# 发布 program_validated 记录
python scripts/publish_rag_knowledge.py --include 公园设计规范 --execute
```

## 国内多模态模型配置

Task 2/3 默认使用阿里云百炼的 OpenAI-compatible 接口；未配置 Key 时仍走规则兜底，
不会发起远端请求。推荐先用 `qwen3.7-plus` 验证空间理解与结构化输出质量，场景稳定后
可评测 `qwen3.6-flash` 以降低成本。

```dotenv
LLM_API_KEY=你的百炼API_KEY
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.7-plus
```

当前范围仅使用多模态模型辅助 Task 2/3，不负责七项形态基线识别，也不调用 Task 4。

## 七项体感请求示例

```json
{
  "comfort": 4.0,
  "naturalness": 4.2,
  "safety": 4.0,
  "relaxation": 4.2,
  "environmental_disturbance": 2.0,
  "stay_intention": 4.0,
  "overall_impression": 4.0
}
```
