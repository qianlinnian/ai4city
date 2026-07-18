"""
================================================================================
Gradio 前端（人机协同主界面）v2
文件: app/gradio_app.py
--------------------------------------------------------------------------------
【流程】
  Step0 下拉选择 assets 图片 → 读 filled_metrics.xlsx + 三张分析图
  Step1 展示情景/形态/九人体感基线
  Step2 调节七个体验目标滑块 → 翻译官
  Step3 人工干预形态要素 → 制图员
  Step4 人工润色自然语言方案 → Seedream 文生图 + 质检
  Step5 修改后多人体验 → 知识库

【启动】
  python app/gradio_app.py
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    ASSETS_DIR,
    EXPERIENCE_KEYS,
    EXPERIENCE_LABELS_ZH,
    FILLED_METRICS_XLSX,
    MORPH_KEYS,
    MORPH_LABELS_ZH,
    TARGET_IMG_DIR,
)
from pipeline.orchestrator import PipelineOrchestrator
from utils.scene_data import list_scene_choices, load_scene_bundle

pipe = PipelineOrchestrator(force_metrics_fallback=True)
SESSION: dict = {"state": None, "bundle": None}

DEFAULT_POST_EDIT_JSON = json.dumps(
    [
        {
            "person_id": "p1",
            "person_name": "参与者A",
            "experience": {
                key: (2 if key == "environmental_disturbance" else 4)
                for key in EXPERIENCE_KEYS
            },
        },
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
            lines.append(f"- {label}: {float(v) * 100:.2f}%")
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


def _fmt_persons(persons: list[dict]) -> str:
    if not persons:
        return "(无)"
    lines = []
    for p in persons:
        name = p.get("person_name") or p.get("person_id")
        exp = p.get("experience") or {}
        parts = [
            f"{EXPERIENCE_LABELS_ZH[k]}={float(exp.get(k, 3)):.0f}"
            for k in EXPERIENCE_KEYS
        ]
        lines.append(f"- **{name}**：{', '.join(parts)}")
    return "\n".join(lines)


def refresh_image_choices():
    choices = list_scene_choices()
    return gr.update(choices=choices, value=(choices[0] if choices else None))


def on_select_image(image_name: str):
    """下拉选图 → 读 Excel + 分析图，填充前端。"""
    empty_exp = [3.0] * len(EXPERIENCE_KEYS)
    empty_morph = _metrics_to_sliders({k: 0 for k in MORPH_KEYS})
    blank = (
        None,  # original
        None,  # edge
        None,  # seg
        None,  # skyline
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "请选择图片",
        "[]",
        *empty_exp,
        *empty_morph,
    )
    if not image_name:
        return blank

    try:
        bundle = load_scene_bundle(image_name)
    except Exception as e:
        raise gr.Error(f"加载场景失败: {e}") from e

    SESSION["bundle"] = bundle
    scene = bundle["scene_context"]
    images = bundle["images"]
    persons = bundle["persons"]
    morph = bundle["morph_metrics"]
    avg = bundle["experience_average"]

    # 目标滑块初值：在均值基础上略抬高正向指标、略降低干扰感
    target_defaults = []
    for k in EXPERIENCE_KEYS:
        v = float(avg.get(k, 3))
        if k == "environmental_disturbance":
            target_defaults.append(max(1.0, min(5.0, round(v - 0.5))))
        else:
            target_defaults.append(max(1.0, min(5.0, round(min(5.0, v + 0.5)))))

    info = (
        f"### 场景 `{bundle.get('scene_id') or ''}` · Excel 行 {bundle.get('excel_row')}\n"
        f"匹配键（前26位）: `{bundle['image_key']}`\n"
        f"指标表: `{FILLED_METRICS_XLSX}`\n"
        f"原图: `{images.get('original') or '（assets 中未找到，请放入原图后再文生图）'}`\n\n"
        f"### 形态要素基线（来自 Excel J–P）\n{_fmt_metrics(morph)}\n\n"
        f"### 九人体感基线\n{_fmt_persons(persons)}"
    )

    return (
        images.get("original"),
        images.get("edge_map"),
        images.get("seg_map"),
        images.get("skyline_map"),
        scene.get("observation_time", ""),
        scene.get("observation_weather", ""),
        scene.get("people_flow", ""),
        scene.get("space_type", ""),
        scene.get("sound_type", ""),
        scene.get("maintenance_status", ""),
        scene.get("traffic_flow", ""),
        "",
        info,
        json.dumps(persons, ensure_ascii=False, indent=2),
        *target_defaults,
        *_metrics_to_sliders(morph),
    )


def step_parse(
    image_name,
    observation_time,
    observation_weather,
    people_flow,
    space_type,
    sound_type,
    maintenance_status,
    traffic_flow,
    scene_desc,
    pre_edit_json,
):
    if not image_name:
        raise gr.Error("请先在下拉列表中选择一张图片")

    try:
        bundle = load_scene_bundle(image_name)
        SESSION["bundle"] = bundle
    except Exception as e:
        raise gr.Error(f"加载场景失败: {e}") from e

    # 若用户改过情景字段，以表单为准
    scene = {
        "observation_time": observation_time or "",
        "observation_weather": observation_weather or "",
        "people_flow": people_flow or "",
        "space_type": space_type or "",
        "sound_type": sound_type or "",
        "maintenance_status": maintenance_status or "",
        "traffic_flow": traffic_flow or "",
        "description": scene_desc or "",
    }

    try:
        pre_edit = _parse_person_experience(pre_edit_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise gr.Error(f"修改前多人体验 JSON 无效: {e}") from e

    if not pre_edit:
        pre_edit = bundle.get("persons") or []

    images = bundle.get("images") or {}
    original = images.get("original")
    if not original:
        # 无原图时仍可用 Excel 指标推进翻译/制图；文生图阶段再校验
        # 用边缘图占位路径写入 session，避免 Path 报错
        placeholder = images.get("edge_map") or images.get("seg_map")
        if not placeholder:
            raise gr.Error(
                f"assets 中找不到原图，且无分析图可用。请将原图放入: {ASSETS_DIR}"
            )
        original = placeholder

    state = pipe.start_session(
        original,
        scene_context=scene,
        pre_edit_experience=pre_edit,
        baseline_metrics=bundle["morph_metrics"],
        image_name=Path(images["original"]).name if images.get("original") else image_name,
        skip_extract=True,
    )
    SESSION["state"] = state

    record_count = len(state.get("pre_edit_experience") or [])
    exp_info = (
        f"已加载 {record_count} 名参与者的逐人评分；翻译时完整输入，不求平均。"
        if record_count
        else "尚未提供逐人评分；翻译前请补充至少一名参与者的完整七项评分。"
    )
    info = (
        f"### 情景要素\n{state['scene_context_text'] or '(未填写)'}\n\n"
        f"### 修改前多人体验记录\n{exp_info}\n\n"
        f"### 形态要素基线（Excel）\n{_fmt_metrics(state['baseline_metrics'])}\n\n"
        f"session_id: `{state['session_id']}`\n\n"
        f"请调节右侧体验目标滑块后点击「确认体验滑块 → 翻译官」。"
    )
    return info


def step_translate(*experience_values):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成「确认加载场景」")

    targets = dict(zip(EXPERIENCE_KEYS, experience_values))
    try:
        state = pipe.run_translator(state, targets)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    SESSION["state"] = state

    tr = state["morph_translation"]
    exp_base = tr.get("experience_baseline", {})
    exp_tgt = tr.get("experience_targets", {})
    exp_lines = "\n".join(
        f"- {EXPERIENCE_LABELS_ZH[k]}: {exp_base.get(k, 3)} → {exp_tgt.get(k, 3)}"
        for k in EXPERIENCE_KEYS
    )
    basis_lines = []
    for item in tr.get("conversion_basis") or []:
        reference = f"（{item.get('reference_id')}）" if item.get("reference_id") else ""
        score = (
            f"，相似度 {float(item['score']):.3f}"
            if item.get("score") is not None
            else ""
        )
        basis_lines.append(
            f"- {item.get('method', 'unknown')}{reference}{score}: {item.get('summary', '')}"
        )
    info = (
        f"### 翻译官：体验变化\n{exp_lines}\n\n"
        f"### 原先形态要素\n{_fmt_metrics(tr.get('baseline_metrics'))}\n\n"
        f"### 目标形态要素（可下方滑块修改）\n{_fmt_metrics(tr.get('target_metrics'))}\n\n"
        f"### 运行方式\n{tr.get('rationale', '')}\n\n"
        f"### 转换依据\n{chr(10).join(basis_lines) or '- 规则兜底'}\n\n"
        f"学习Agent: {'已参考' if tr.get('learning_applied') else '未启用（占位）'}"
    )
    sliders = _metrics_to_sliders(state["confirmed_target_metrics"])
    return info, *sliders


def _fmt_layout_plan(plan: dict) -> str:
    actions = []
    for item in plan.get("object_actions") or []:
        actions.append(
            f"- **{item.get('action', 'adjust')} {item.get('object_type', '空间对象')}**："
            f"{item.get('position', '')}；{item.get('quantity', '')}"
        )
    return (
        f"### 空间布局方案\n{plan.get('plan_summary', '')}\n\n"
        f"### 对象级修改\n{chr(10).join(actions) or '- 轻量优化现有空间对象'}\n\n"
        f"### 保持不变区域\n"
        + "\n".join(f"- {x}" for x in plan.get("unchanged_regions") or [])
        + f"\n\n### 生成依据\n{plan.get('rationale', '')}"
    )


def step_confirm_morph(expert_advice, *slider_vals):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成「确认体验滑块 → 翻译官」")
    human_metrics = _sliders_to_metrics(list(slider_vals))
    note = (expert_advice or "").strip() or "前端人工确认形态要素"
    state = pipe.confirm_morph(
        state, human_metrics=human_metrics, note=note, language="zh"
    )
    SESSION["state"] = state
    plan = state.get("modification_plan") or {}
    return plan.get("draft_text", ""), _fmt_layout_plan(plan)


def step_generate(final_plan):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成前面步骤")
    if not final_plan or not final_plan.strip():
        raise gr.Error("修改方案不能为空")

    bundle = SESSION.get("bundle") or {}
    original = (bundle.get("images") or {}).get("original")
    if not original or not Path(original).is_file():
        raise gr.Error(
            f"Seedream 需要 assets 原图。请将与所选场景对应的全景图放入:\n{ASSETS_DIR}"
        )
    # 确保 session 使用真实原图
    state = dict(state)
    state["image_path"] = original
    state["image_name"] = Path(original).name

    state = pipe.confirm_plan(state, final_plan)
    state = pipe.generate_and_check(state)
    SESSION["state"] = state

    gen = state["generation"]
    qr = state.get("quality_report") or {}
    out_path = gen.get("output_image_path")
    if out_path:
        out_path = str(Path(out_path).resolve())
        if not Path(out_path).is_file():
            raise gr.Error(f"生成文件不存在，无法展示: {out_path}")

    is_mock = bool(gen.get("mock"))
    fallback_err = (gen.get("raw") or {}).get("fallback_error")
    mode_line = (
        f"### 生成模式: **MOCK 演示**（非真实 Seedream）\n"
        f"回退原因: `{fallback_err or (gen.get('raw') or {}).get('note', '')}`\n"
        if is_mock
        else "### 生成模式: **Seedream 实网 API**\n"
    )
    report = (
        f"{mode_line}"
        f"### 输出路径: `{out_path}`（目录 `{TARGET_IMG_DIR}`）\n"
        f"### 质检: {'通过' if qr.get('passed') else '未通过'}\n"
        f"{qr.get('details', '')}\n\n"
        f"### 实测形态要素\n{_fmt_metrics(qr.get('measured_metrics'))}\n\n"
        f"### 目标形态要素\n{_fmt_metrics(qr.get('target_metrics'))}\n\n"
        f"### 偏差\n```json\n{json.dumps(qr.get('deviations'), ensure_ascii=False, indent=2)}\n```"
    )
    measured_sliders = _metrics_to_sliders(qr.get("measured_metrics") or {})
    return out_path, report, *measured_sliders


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
    choices = list_scene_choices()
    with gr.Blocks(title="高密度微空间 · 多智能体优化 v2") as demo:
        gr.Markdown(
            """
            # 高密度情境下微空间优化 · 人机多智能体流水线 v2
            **选图(Excel指标+分析图) → 体验滑块 → 翻译官 → 制图员 → Seedream 文生图 → 反馈入库**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_dropdown = gr.Dropdown(
                    choices=choices,
                    value=(choices[0] if choices else None),
                    label=f"选择场景图片（优先 {ASSETS_DIR.name}/，否则边缘图 stem）",
                    interactive=True,
                )
                btn_refresh = gr.Button("刷新图片列表")
                original_img = gr.Image(label="原图（assets）", type="filepath", height=220)
                with gr.Row():
                    edge_img = gr.Image(label="边缘密度 edge_", type="filepath", height=160)
                    seg_img = gr.Image(label="语义分割 seg_", type="filepath", height=160)
                    skyline_img = gr.Image(label="天际线 skyline_", type="filepath", height=160)

                gr.Markdown("### 情景要素（来自 Excel C–I，可改）")
                observation_time = gr.Textbox(label="观测时间")
                observation_weather = gr.Textbox(label="观测天气")
                people_flow = gr.Textbox(label="人流量")
                space_type = gr.Textbox(label="空间类型（社区、蓝绿、商办）")
                sound_type = gr.Textbox(label="声音类型")
                maintenance_status = gr.Textbox(label="管理维护状态")
                traffic_flow = gr.Textbox(label="交通流量")
                scene_desc = gr.Textbox(label="补充描述", lines=2)
                pre_edit_json = gr.Textbox(
                    label="修改前多人体验（JSON，自动从 Excel 九人填入）",
                    value="[]",
                    lines=10,
                )
                btn_parse = gr.Button("① 确认加载场景 → 写入 Session", variant="primary")

            with gr.Column(scale=1):
                parse_md = gr.Markdown("请先选择图片…")
                gr.Markdown("### Step2 七项体验感受目标（1–5；环境干扰感越低越好）")
                experience_sliders = [
                    gr.Slider(1, 5, value=3, step=1, label=EXPERIENCE_LABELS_ZH[key])
                    for key in EXPERIENCE_KEYS
                ]
                btn_translate = gr.Button("② 确认体验滑块 → 翻译官", variant="primary")

        translate_md = gr.Markdown("翻译官结果将显示在这里")
        gr.Markdown("### 人工干预①：形态要素目标（初值=Excel 基线，翻译后会被更新）")
        morph_sliders = []
        for k in MORPH_KEYS:
            if k == "color_richness":
                morph_sliders.append(
                    gr.Slider(0, 24, value=8, step=0.1, label=MORPH_LABELS_ZH[k])
                )
            else:
                morph_sliders.append(
                    gr.Slider(0, 100, value=10, step=0.1, label=f"{MORPH_LABELS_ZH[k]} (%)")
                )
        expert_advice = gr.Textbox(
            label="专家建议（可选，将传给制图员 Agent）",
            placeholder="例如：保持左侧历史建筑立面不变，优先优化右侧停留空间",
            lines=2,
        )
        btn_morph = gr.Button(
            "③ 确认形态要素 → 制图员生成空间布局方案", variant="primary"
        )

        plan_box = gr.Textbox(label="人工干预②：自然语言修改方案（可润色）", lines=8)
        plan_rationale = gr.Markdown("")
        btn_gen = gr.Button("④ 确认方案 → Seedream 文生图 + 质检", variant="primary")

        with gr.Row():
            out_img = gr.Image(label="修改后全景（TargetIMG）", type="filepath", height=320)
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
                    gr.Slider(0, 24, value=8, step=0.1, label=f"纠偏-{MORPH_LABELS_ZH[k]}")
                )
            else:
                corrected_sliders.append(
                    gr.Slider(
                        0, 100, value=10, step=0.1, label=f"纠偏-{MORPH_LABELS_ZH[k]} (%)"
                    )
                )

        with gr.Row():
            score = gr.Slider(1, 5, value=4, step=1, label="方案主观评分")
            notes = gr.Textbox(label="备注", placeholder="例如：西晒墙应改耐旱植物…")
        btn_mem = gr.Button("⑤ 沉淀到本地知识库", variant="primary")
        mem_out = gr.Textbox(label="知识库写入结果")

        select_outputs = [
            original_img,
            edge_img,
            seg_img,
            skyline_img,
            observation_time,
            observation_weather,
            people_flow,
            space_type,
            sound_type,
            maintenance_status,
            traffic_flow,
            scene_desc,
            parse_md,
            pre_edit_json,
            *experience_sliders,
            *morph_sliders,
        ]

        btn_refresh.click(refresh_image_choices, outputs=[image_dropdown])
        image_dropdown.change(on_select_image, inputs=[image_dropdown], outputs=select_outputs)
        demo.load(on_select_image, inputs=[image_dropdown], outputs=select_outputs)

        btn_parse.click(
            step_parse,
            inputs=[
                image_dropdown,
                observation_time,
                observation_weather,
                people_flow,
                space_type,
                sound_type,
                maintenance_status,
                traffic_flow,
                scene_desc,
                pre_edit_json,
            ],
            outputs=[parse_md],
        )
        btn_translate.click(
            step_translate,
            inputs=experience_sliders,
            outputs=[translate_md, *morph_sliders],
        )
        btn_morph.click(
            step_confirm_morph,
            inputs=[expert_advice, *morph_sliders],
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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
