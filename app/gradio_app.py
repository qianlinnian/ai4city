"""
================================================================================
Gradio 前端（人机协同主界面）v2
文件: app/gradio_app.py
--------------------------------------------------------------------------------
【新流程】
  Step0 上传全景 JPG + 情景要素 +（可选）修改前多人体验
  Step1 形态解析 → 展示 7 维形态基线
  Step2 调节五个体验滑块 → 确认 → 翻译官（体验→形态目标）
  Step3 人工干预形态要素目标 → 确认 → 制图员（自然语言方案）
  Step4 人工干预修改方案文本 → 确认 → World Labs 文生图 + 质检
  Step5 填写修改后多人体验 → 沉淀知识库

【启动】
  cd code
  python app/gradio_app.py
================================================================================
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EXPERIENCE_KEYS, EXPERIENCE_LABELS_ZH, MORPH_KEYS, MORPH_LABELS_ZH, UPLOAD_DIR
from pipeline.orchestrator import PipelineOrchestrator

pipe = PipelineOrchestrator(force_metrics_fallback=True)
SESSION: dict = {"state": None}

DEFAULT_PRE_EDIT_JSON = json.dumps(
    [
        {"person_id": "p1", "person_name": "参与者A", "experience": {"comfort": 3, "restoration": 3, "safety": 3, "pleasure": 3, "stay": 3}},
        {"person_id": "p2", "person_name": "参与者B", "experience": {"comfort": 3, "restoration": 3, "safety": 3, "pleasure": 3, "stay": 3}},
    ],
    ensure_ascii=False,
    indent=2,
)

DEFAULT_POST_EDIT_JSON = json.dumps(
    [
        {"person_id": "p1", "person_name": "参与者A", "experience": {"comfort": 4, "restoration": 4, "safety": 4, "pleasure": 4, "stay": 4}},
        {"person_id": "p2", "person_name": "参与者B", "experience": {"comfort": 4, "restoration": 4, "safety": 4, "pleasure": 4, "stay": 4}},
    ],
    ensure_ascii=False,
    indent=2,
)


def _fmt_metrics(d: dict | None) -> str:
    if not d:
        return "(空)"
    lines = []
    for k in MORPH_KEYS:
        if k not in d:
            continue
        v = d[k]
        label = MORPH_LABELS_ZH.get(k, k)
        if k == "color_richness":
            lines.append(f"- {label}: {float(v):.2f}")
        else:
            lines.append(f"- {label}: {float(v)*100:.2f}%")
    return "\n".join(lines)


def _metrics_to_sliders(d: dict) -> list[float]:
    vals = []
    for k in MORPH_KEYS:
        v = float(d.get(k, 0))
        if k == "color_richness":
            vals.append(v)
        else:
            vals.append(round(v * 100, 2))
    return vals


def _sliders_to_metrics(values: list[float]) -> dict:
    out = {}
    for k, v in zip(MORPH_KEYS, values):
        if k == "color_richness":
            out[k] = float(v)
        else:
            out[k] = float(v) / 100.0
    return out


def _parse_person_experience(raw: str) -> list[dict]:
    if not raw or not raw.strip():
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("多人体验须为 JSON 数组")
    return data


def step_parse(image, location_type, time_of_day, weather, crowd_level, scene_desc, pre_edit_json):
    if image is None:
        raise gr.Error("请先上传全景图")
    src = Path(image)
    dest = UPLOAD_DIR / src.name
    shutil.copy(src, dest)

    try:
        pre_edit = _parse_person_experience(pre_edit_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise gr.Error(f"修改前多人体验 JSON 无效: {e}") from e

    scene = {
        "location_type": location_type or "",
        "time_of_day": time_of_day or "",
        "weather": weather or "",
        "crowd_level": crowd_level or "",
        "description": scene_desc or "",
    }
    state = pipe.start_session(dest, scene_context=scene, pre_edit_experience=pre_edit)
    SESSION["state"] = state

    exp_base = state["experience_baseline"]
    exp_info = "\n".join(
        f"- {EXPERIENCE_LABELS_ZH[k]}: {exp_base[k]}" for k in EXPERIENCE_KEYS
    )
    info = (
        f"### 情景要素\n{state['scene_context_text'] or '(未填写)'}\n\n"
        f"### 修改前多人体验均值（作为体验原值参考）\n{exp_info}\n\n"
        f"### 形态要素基线（图像解析）\n{_fmt_metrics(state['baseline_metrics'])}\n\n"
        f"session_id: `{state['session_id']}`"
    )
    return info


def step_translate(comfort, restoration, safety, pleasure, stay):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成「解析全景」")
    if state.get("stage") not in ("await_experience_confirm", "await_morph_confirm"):
        pass  # 允许重新调节

    targets = {
        "comfort": comfort,
        "restoration": restoration,
        "safety": safety,
        "pleasure": pleasure,
        "stay": stay,
    }
    state = pipe.run_translator(state, targets)
    SESSION["state"] = state

    tr = state["morph_translation"]
    exp_base = tr.get("experience_baseline", {})
    exp_tgt = tr.get("experience_targets", {})
    exp_lines = "\n".join(
        f"- {EXPERIENCE_LABELS_ZH[k]}: {exp_base.get(k,3)} → {exp_tgt.get(k,3)}"
        for k in EXPERIENCE_KEYS
    )
    info = (
        f"### 翻译官：体验变化\n{exp_lines}\n\n"
        f"### 原先形态要素\n{_fmt_metrics(tr.get('baseline_metrics'))}\n\n"
        f"### 目标形态要素（可下方滑块修改）\n{_fmt_metrics(tr.get('target_metrics'))}\n\n"
        f"### 翻译理由\n{tr.get('rationale', '')}\n\n"
        f"学习Agent: {'已参考' if tr.get('learning_applied') else '未启用（占位）'}"
    )
    sliders = _metrics_to_sliders(state["confirmed_target_metrics"])
    return info, *sliders


def step_confirm_morph(*slider_vals):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成「确认体验滑块 → 翻译官」")
    human_metrics = _sliders_to_metrics(list(slider_vals))
    state = pipe.confirm_morph(state, human_metrics=human_metrics, note="前端人工确认形态要素")
    SESSION["state"] = state
    plan = state.get("modification_plan") or {}
    return plan.get("draft_text", ""), plan.get("rationale", "")


def step_generate(final_plan):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成前面步骤")
    if not final_plan or not final_plan.strip():
        raise gr.Error("修改方案不能为空")

    state = pipe.confirm_plan(state, final_plan)
    state = pipe.generate_and_check(state)
    SESSION["state"] = state

    gen = state["generation"]
    qr = state["quality_report"]
    report = (
        f"### 生成模式: {'MOCK 演示' if gen.get('mock') else 'World Labs API'}\n"
        f"### 质检: {'通过' if qr.get('passed') else '未通过'}\n"
        f"{qr.get('details', '')}\n\n"
        f"### 实测形态要素\n{_fmt_metrics(qr.get('measured_metrics'))}\n\n"
        f"### 目标形态要素\n{_fmt_metrics(qr.get('target_metrics'))}\n\n"
        f"### 偏差\n```json\n{json.dumps(qr.get('deviations'), ensure_ascii=False, indent=2)}\n```"
    )
    measured_sliders = _metrics_to_sliders(qr.get("measured_metrics") or {})
    return gen.get("output_image_path"), report, *measured_sliders


def step_save_memory(score, notes, post_edit_json, *corrected_sliders):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("无会话")
    try:
        post_edit = _parse_person_experience(post_edit_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise gr.Error(f"修改后多人体验 JSON 无效: {e}") from e

    state = pipe.record_post_experience(state, post_edit)
    corrected = _sliders_to_metrics(list(corrected_sliders))
    state = pipe.save_memory(
        state,
        human_corrected_metrics=corrected,
        score=float(score) if score is not None else None,
        notes=notes or "",
    )
    SESSION["state"] = state
    learning_stats = pipe.learning.get_stats()
    return (
        f"已写入知识库，memory_id = {state['memory_id']}，session = {state['session_id']}\n"
        f"学习Agent统计: {json.dumps(learning_stats, ensure_ascii=False)}"
    )


def build_ui():
    with gr.Blocks(title="高密度微空间 · 多智能体优化 v2", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 高密度情境下微空间优化 · 人机多智能体流水线 v2
            **全景+情景 → 形态解析 → 体验滑块 → 翻译官(体验→形态) → 制图员(形态→文本) → 文生图 → 多人体验反馈 → 知识库**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="上传全景 JPG/PNG", height=260)
                gr.Markdown("### 情景要素")
                location_type = gr.Textbox(label="空间类型", placeholder="街巷 / 广场 / 口袋公园")
                time_of_day = gr.Textbox(label="时段", placeholder="清晨 / 午后 / 傍晚")
                weather = gr.Textbox(label="天气", placeholder="晴 / 阴 / 雨")
                crowd_level = gr.Textbox(label="人流密度", placeholder="稀疏 / 中等 / 拥挤")
                scene_desc = gr.Textbox(label="补充描述", lines=2)
                pre_edit_json = gr.Textbox(
                    label="修改前多人体验（JSON，可选）",
                    value=DEFAULT_PRE_EDIT_JSON,
                    lines=8,
                )
                btn_parse = gr.Button("① 解析全景 + 形态要素", variant="primary")

            with gr.Column(scale=1):
                parse_md = gr.Markdown("等待解析…")
                gr.Markdown("### Step2 体验感受目标（五个滑块，确认后送翻译官）")
                comfort = gr.Slider(1, 5, value=4, step=1, label=EXPERIENCE_LABELS_ZH["comfort"])
                restoration = gr.Slider(1, 5, value=5, step=1, label=EXPERIENCE_LABELS_ZH["restoration"])
                safety = gr.Slider(1, 5, value=3, step=1, label=EXPERIENCE_LABELS_ZH["safety"])
                pleasure = gr.Slider(1, 5, value=4, step=1, label=EXPERIENCE_LABELS_ZH["pleasure"])
                stay = gr.Slider(1, 5, value=4, step=1, label=EXPERIENCE_LABELS_ZH["stay"])
                btn_translate = gr.Button("② 确认体验滑块 → 翻译官", variant="primary")

        translate_md = gr.Markdown("翻译官结果将显示在这里")
        gr.Markdown("### 人工干预①：形态要素目标")
        morph_sliders = []
        for k in MORPH_KEYS:
            if k == "color_richness":
                morph_sliders.append(gr.Slider(1, 12, value=3.5, step=0.1, label=MORPH_LABELS_ZH[k]))
            else:
                morph_sliders.append(
                    gr.Slider(0, 80, value=15, step=0.1, label=f"{MORPH_LABELS_ZH[k]} (%)")
                )
        btn_morph = gr.Button("③ 确认形态要素 → 制图员生成方案", variant="primary")

        plan_box = gr.Textbox(label="人工干预②：自然语言修改方案（可润色）", lines=8)
        plan_rationale = gr.Markdown("")
        btn_gen = gr.Button("④ 确认方案 → World Labs 文生图 + 质检", variant="primary")

        with gr.Row():
            out_img = gr.Image(label="修改后全景", type="filepath", height=320)
            quality_md = gr.Markdown("质检报告将显示在这里")

        gr.Markdown("### 修改后多人体验 + 指标纠偏（写入知识库前）")
        post_edit_json = gr.Textbox(
            label="修改后多人体验（JSON）",
            value=DEFAULT_POST_EDIT_JSON,
            lines=8,
        )
        corrected_sliders = []
        for k in MORPH_KEYS:
            if k == "color_richness":
                corrected_sliders.append(
                    gr.Slider(1, 12, value=3.5, step=0.1, label=f"纠偏-{MORPH_LABELS_ZH[k]}")
                )
            else:
                corrected_sliders.append(
                    gr.Slider(0, 80, value=15, step=0.1, label=f"纠偏-{MORPH_LABELS_ZH[k]} (%)")
                )

        with gr.Row():
            score = gr.Slider(1, 5, value=4, step=1, label="方案主观评分")
            notes = gr.Textbox(label="备注", placeholder="例如：西晒墙应改耐旱植物…")
        btn_mem = gr.Button("⑤ 沉淀到本地知识库", variant="primary")
        mem_out = gr.Textbox(label="知识库写入结果")

        btn_parse.click(
            step_parse,
            inputs=[image, location_type, time_of_day, weather, crowd_level, scene_desc, pre_edit_json],
            outputs=[parse_md],
        )
        btn_translate.click(
            step_translate,
            inputs=[comfort, restoration, safety, pleasure, stay],
            outputs=[translate_md, *morph_sliders],
        )
        btn_morph.click(
            step_confirm_morph,
            inputs=morph_sliders,
            outputs=[plan_box, plan_rationale],
        )
        btn_gen.click(
            step_generate,
            inputs=[plan_box],
            outputs=[out_img, quality_md, *corrected_sliders],
        )
        btn_mem.click(
            step_save_memory,
            inputs=[score, notes, post_edit_json, *corrected_sliders],
            outputs=[mem_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
