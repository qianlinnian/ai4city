# AI4City MAS / Urban Evidence Engine

这是依据当前六份项目文档整理出的多智能体代码骨架。目标是把高密度城市微空间研究组织为一条可追踪的证据链：

`全景/视频/元数据 -> 感知描述 -> 特征 -> 专家校准 -> 情境矩阵 -> AI评分 -> 干预方案 -> VR实验 -> 人因验证 -> 参数化策略 -> 证据包`

当前版本是 **backbone**，不是已经训练完成的评分模型。默认 `mock` 适配器会完整跑通 11 个阶段并生成结构化产物；真实多模态模型、图像重绘、生理传感和 Grasshopper 通过适配器接入。

## 快速运行

仓库提供的 PowerShell 入口会自动定位 `ai4city-mas` Conda 环境，规避当前机器上 `conda run` 误命中 MSYS2 Python 的问题。

```powershell
.\scripts\ai4city.ps1 plan
.\scripts\ai4city.ps1 run --config configs\default.yaml
.\scripts\ai4city.ps1 status runs\<run-id>
.\scripts\ai4city.ps1 test
```

## 目录

- `src/ai4city_mas/`: 领域模型、智能体、适配器和 LangGraph 编排。
- `configs/default.yaml`: 18 种 `Time x Crowd x Noise` 情境及实验门控配置。
- `examples/demo_dataset/`: 无隐私的最小演示数据。
- `docs/UNIFIED_PIPELINE.md`: 文档归并后的统一 pipeline、职责和数据契约。
- `stitch/`: 现有 UI 视觉原型，仅作 Dashboard 参考，尚未连接运行时。
- `runs/`: 每次运行的 manifest、阶段产物和最终 evidence package，不进入 Git。

## 设计原则

- 专家校准和 VR 伦理审批是显式门控，不是备注。
- 没有真实 VR/生理数据时，不生成虚假的统计显著性或阈值。
- 每个阶段只通过版本化 JSON/CSV 数据契约衔接，便于替换模型。
- 原始参与者标识、面部、车牌和敏感元数据不得进入公开产物。
