# Unified AI4City Pipeline

## 目标

系统研究的是动态情境与静态形态如何共同影响微空间体验，并把经验证的关系转译成可执行的设计参数。AI 负责高通量识别、推演和生成；规划/心理/人因专家负责问题定义、边界条件、异常解释和最终裁决。

## 统一的 11 个阶段

| 阶段 | Owner | 核心动作 | 标准产物 |
|---|---|---|---|
| 01 数据加载 | Pipeline | 校验全景、短视频、元数据和隐私状态 | `scene_index.json` |
| 02 感知描述 | E-Agent | 多模态模型生成结构化生态/社会/空间描述 | `perception_descriptions.json` |
| 03 特征提取 | E-Agent | 语义嵌入、分割和 GVF/SVF 等形态特征 | `feature_matrix.json` |
| 04 专家校准 | Expert gate | 修正描述、权重、标签和设计边界，保留理由 | `expert_calibration.json` |
| 05 情境矩阵 | C-Agent | 生成 Time x Crowd x Noise 组合及场景假设 | `experiment_matrix.json` |
| 06 感知评分 | C-Agent | 输出多维预测及不确定性，形成 AI 基线 | `ai_scores.json` |
| 07 干预刺激 | D-Agent | 生成受结构约束的干预梯度和 VR 刺激计划 | `intervention_catalog.json` |
| 08 VR 实验 | X-Agent | 场景编排、事件标记、主观/HRV/EDA 数据接入 | `vr_experiment_plan.json` |
| 09 人因验证 | H-Agent | 清洗对齐、模型比较、偏差诊断和阈值估计 | `validation_report.json` |
| 10 参数化优化 | D-Agent | 将有效作用域、风险边界转为参数包/帕累托方案 | `parametric_strategy.json` |
| 11 证据汇报 | Pipeline | 汇总来源、模型、人工修订、验证和风险提示 | `evidence_package.json` |

## 五类智能体

- **E-Agent / Environment**：把图像、视频和元数据转成可审阅的环境描述与特征。
- **C-Agent / Context**：构造动态情境，形成场景-情境组合的预测基线。
- **X-Agent / Experiment**：执行 VR 暴露流程、时间同步、安全暂停和事件记录。
- **H-Agent / Human factors**：分析主观和生理数据，定位 AI 偏差，估计作用域与阈值。
- **D-Agent / Design**：生成干预样本，并把验证结果转译为参数约束和设计方案。

## 人工门控

1. **专家校准门控**：阶段 04 未批准时，不能进入情境推演。所有修改必须记录原值、新值、操作者、理由和时间。
2. **VR/伦理门控**：阶段 08 未获伦理与安全许可时，只能生成实验草案，不能采集参与者数据。
3. **设计裁决**：专家可否决统计显著但实际意义弱、不可施工或有安全风险的策略；系统保留冲突记录。

## 核心数据契约

- 所有记录包含 `run_id`、`scene_id`、`schema_version` 和来源信息。
- AI 评分必须同时给出 `model_id`、`prompt_version`、`score` 和 `confidence`。
- 人工校准不得覆盖原始输出，只能追加 revision。
- 阈值只有在真实观察数据达到最小样本和质量标准后才能标记为 `validated`。
- 参数化导出区分 `hypothesis`、`expert_supported`、`empirically_validated` 三种证据等级。

## 实施边界

当前 backbone 已实现编排、门控、产物追踪和可重复 mock 运行。以下能力通过适配器逐步接入：多模态 VLM/分割模型、真实评分模型、ControlNet 全景重绘、VR/LSL/BIOPAC、NeuroKit2/统计模型、Rhino/Grasshopper/MCP、Dashboard API。

文档中出现的 42% GVF、22%-46% 有效作用域等数值属于示例性研究假设，不能作为默认硬约束。系统只有在阶段 09 得到真实验证后才把阈值传给阶段 10。

## 六份文档的归并决策

- `AI4Cities夏令营学员任务书0709.docx` 作为最新执行口径：3-5 个核心场景、11 步任务链、VR 低负担验证、七项主观指标和最终成果要求。
- `工作营完整技术 Pipeline20260615.docx` 提供 11 个阶段的技术接口与产物命名，是 pipeline 骨架的主要来源。
- `夏令营引入领域专家的人机多智能体协同工作营.docx` 提供 E/C/X/H/D 五类智能体和 Expert-in-the-Loop 机制。
- `高密度情境下微空间优化参数和应用场景.docx` 提供参数接口与阈值传导示例；其中数值按研究假设处理，不能直接固化。
- `高密度情境下微空间优化科学问题阐述.docx` 确定科学边界：研究情境-形态耦合和可解释反向设计，不是用 AI 替代问卷。
- `高密度情境下微空间优化-AI4Cities工作营.docx` 提供六天教学叙事、VR/传感和 Grasshopper 扩展场景。

文档间存在三处口径差异，代码采用兼容性处理：评分维度从早期四项扩展为任务书的七项；VR 生理数据从“必需”降为可选适配器；样本规模同时支持教学演示的 3-5 个场景和批处理的 50+ 场景。具体日历日期不写死在 pipeline，由活动配置单独管理。
