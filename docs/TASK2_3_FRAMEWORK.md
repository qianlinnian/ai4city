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
4. 当前 RAG 不执行检索，只通过 `TranslationRagProvider` 保留未来注入接口。
5. 无 LLM、调用失败或模型输出越界时，才对每位参与者分别应用映射规则，再对形态目标取中位数兜底。
6. 程序计算形态前后差值，并保留专家编辑确认入口。

## Task 3：制图员 Agent

输入：

- 原始 JPG
- 修改前后的七项形态要素值
- 七项体感目标、情景要素和专家建议
- 可选未来 RAG 接口（当前默认关闭）

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
- 多模态链：使用 LangChain `SystemMessage` / `HumanMessage`，把压缩后的全景 JPG 作为 base64 图像内容块输入。
- RAG：Task 2/3 当前均不执行检索；仅保留可注入 Provider 接口，待课本、案例和专家知识准备完成后再启用。
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
