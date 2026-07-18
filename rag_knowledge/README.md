# ai4city 正式 RAG 知识库

- `published/`：唯一会被 Task 2/3 RAG 自动读取的正式知识。
- `last_publish_report.json`：最近一次发布中暂缓记录及原因，不参与检索。
- DeepSeek 草稿仍保存在 `outputs/knowledge_drafts/`，不会直接进入本目录。

程序发布条件统一为记录状态 `program_validated`，其含义是：Schema v2、无校验错误、证据具有真实页码和 OCR 行号、无 missing 证据、置信度不低于 0.75、模糊证据分数不低于 0.85、非高风险规范且模型未要求复核。

`needs_review`、`empty`、没有行号或证据无法定位的记录不发布。项目当前不设置额外专家批准层。
