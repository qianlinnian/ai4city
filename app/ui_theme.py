"""Gradio 工作台的视觉主题与轻量 HTML 组件。"""

from __future__ import annotations

import html

import gradio as gr


APP_THEME = gr.themes.Ocean(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md",
    font=[
        gr.themes.GoogleFont("Inter"),
        "Microsoft YaHei",
        "PingFang SC",
        "sans-serif",
    ],
).set(
    body_background_fill="#f3f7fb",
    body_background_fill_dark="#0b1220",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#dbe6f2",
    block_shadow="0 12px 36px rgba(21, 55, 92, 0.08)",
    button_primary_background_fill="linear-gradient(135deg, #1769e0, #0d9488)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1259c5, #0f766e)",
    button_primary_border_color="transparent",
)


APP_CSS = r"""
.gradio-container {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: 1540px !important;
  margin: 0 auto !important;
  padding: 22px 28px 42px !important;
}

.app-hero {
  position: relative;
  overflow: hidden;
  padding: 30px 34px;
  margin-bottom: 18px;
  border: 1px solid rgba(30, 105, 224, 0.16);
  border-radius: 24px;
  color: #f8fbff;
  background:
    radial-gradient(circle at 85% 15%, rgba(45, 212, 191, .28), transparent 30%),
    linear-gradient(128deg, #102b55 0%, #1557a6 58%, #0f766e 120%);
  box-shadow: 0 18px 50px rgba(16, 43, 85, .18);
}
.app-hero::after {
  content: "";
  position: absolute;
  inset: auto -60px -90px auto;
  width: 260px;
  height: 260px;
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 50%;
}
.app-eyebrow {
  font-size: 12px;
  font-weight: 750;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: #99f6e4;
}
.app-hero h1 {
  margin: 7px 0 8px;
  color: #ffffff !important;
  font-size: clamp(28px, 3vw, 42px);
  line-height: 1.14;
  letter-spacing: -.03em;
}
.app-hero p {
  max-width: 850px;
  margin: 0;
  color: #dbeafe;
  font-size: 15px;
}
.hero-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
.hero-tag {
  display: inline-flex;
  padding: 6px 10px;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px;
  background: rgba(255,255,255,.09);
  color: #ecfeff;
  font-size: 12px;
}

.workflow-stepper {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0 18px;
}
.workflow-step {
  min-height: 64px;
  padding: 12px 14px;
  border: 1px solid #dbe6f2;
  border-radius: 16px;
  background: rgba(255,255,255,.88);
  color: #64748b;
}
.workflow-step .step-number {
  display: inline-grid;
  place-items: center;
  width: 24px;
  height: 24px;
  margin-right: 6px;
  border-radius: 8px;
  background: #e8eef6;
  color: #476078;
  font-weight: 800;
  font-size: 12px;
}
.workflow-step strong { display: block; margin-top: 6px; font-size: 13px; color: #334155; }
.workflow-step.active {
  border-color: #2b78e4;
  background: linear-gradient(145deg, #eff6ff, #ecfeff);
  box-shadow: 0 8px 24px rgba(43,120,228,.11);
}
.workflow-step.active .step-number { background: #1769e0; color: #fff; }
.workflow-step.done { border-color: #99d8d0; background: #f0fdfa; }
.workflow-step.done .step-number { background: #0d9488; color: #fff; }

.section-kicker {
  color: #1769e0;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.section-title { margin: 2px 0 14px; color: #172b4d; font-size: 21px; font-weight: 780; }
.soft-note {
  padding: 10px 12px;
  border-left: 3px solid #14b8a6;
  border-radius: 9px;
  background: #f0fdfa;
  color: #335b62;
  font-size: 13px;
}
.status-strip {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 9px 12px;
  border: 1px solid #dbe6f2;
  border-radius: 12px;
  background: #f8fafc;
  color: #475569;
  font-size: 13px;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; }
.status-dot.ok { background: #0d9488; box-shadow: 0 0 0 4px rgba(13,148,136,.12); }
.status-dot.warn { background: #f59e0b; box-shadow: 0 0 0 4px rgba(245,158,11,.12); }
.status-dot.error { background: #ef4444; box-shadow: 0 0 0 4px rgba(239,68,68,.12); }

.metric-grid table { font-size: 13px !important; }
.metric-grid th { background: #eff6ff !important; color: #254b79 !important; }
.primary-action button { min-height: 46px; font-weight: 760; letter-spacing: .01em; }
.secondary-action button { min-height: 42px; }
.compact-card { min-height: 100%; }
.image-stage img { border-radius: 14px !important; }
.footer-note { margin-top: 20px; text-align: center; color: #7b8da5; font-size: 12px; }

/* Gradio 在深色模式下会切换组件文字颜色；以下规则同步切换自定义卡片，
   避免浅色背景和白色文字叠在一起。 */
body.dark,
body.dark .gradio-container {
  background-color: #0b1220 !important;
  color: #e5edf7 !important;
}
.dark .workflow-step {
  border-color: #2b3d57;
  background: rgba(17, 28, 46, .94);
  color: #aebdd0;
}
.dark .workflow-step .step-number { background: #24344c; color: #cbd5e1; }
.dark .workflow-step strong { color: #e7edf6; }
.dark .workflow-step.active {
  border-color: #4f9cf9;
  background: linear-gradient(145deg, #132b4a, #102f35);
}
.dark .workflow-step.done {
  border-color: #287e75;
  background: #102a2a;
}
.dark .section-kicker { color: #60a5fa; }
.dark .section-title { color: #f1f5f9; }
.dark .soft-note {
  background: #102a2a;
  color: #c7f3ed;
}
.dark .status-strip {
  border-color: #2b3d57;
  background: #111c2e;
  color: #dbe5f1;
}
.dark .metric-grid th {
  background: #172a45 !important;
  color: #dbeafe !important;
}
.dark .metric-grid td { color: #e5edf7 !important; }
.dark .footer-note { color: #aebdd0; }

@media (max-width: 900px) {
  .gradio-container { padding: 12px !important; }
  .app-hero { padding: 24px 20px; border-radius: 18px; }
  .workflow-stepper { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
"""


_STEP_LABELS = (
    "输入与体验",
    "形态目标审核",
    "空间布局方案",
    "生成与验证",
)


def stage_stepper_html(active_step: int = 0) -> str:
    """返回四阶段步骤条；active_step 使用 0~3。"""
    active_step = max(0, min(3, int(active_step)))
    blocks = []
    for index, label in enumerate(_STEP_LABELS):
        status = "done" if index < active_step else "active" if index == active_step else ""
        blocks.append(
            f'<div class="workflow-step {status}">'
            f'<span class="step-number">{index + 1:02d}</span>'
            f'<strong>{html.escape(label)}</strong>'
            "</div>"
        )
    return '<div class="workflow-stepper">' + "".join(blocks) + "</div>"


def status_html(message: str, kind: str = "neutral") -> str:
    css_kind = kind if kind in {"ok", "warn", "error"} else ""
    return (
        '<div class="status-strip">'
        f'<span class="status-dot {css_kind}"></span>'
        f"<span>{html.escape(message)}</span>"
        "</div>"
    )
