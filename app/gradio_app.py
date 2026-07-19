"""
================================================================================
Gradio 前端（人机协同主界面）v2
文件: app/gradio_app.py
--------------------------------------------------------------------------------
【流程】
  Step0 下拉选择 assets 图片 → 读 filled_metrics.xlsx + 三张分析图
  Step1 表格展示九人体感基线 + 情景/形态
  Step2 调节七个体验目标滑块 → 翻译官
  Step3 人工干预形态要素 → 制图员
  Step4 人工润色自然语言方案 → Seedream 文生图 + 原图对比
  Step5 表格填写修改后多人体验 → 知识库

【启动】
  python app/gradio_app.py
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

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

# 体验表格列：姓名 + 七项指标
_EXP_NAME_COL = "姓名"
_EXP_HEADERS = [_EXP_NAME_COL] + [EXPERIENCE_LABELS_ZH[k] for k in EXPERIENCE_KEYS]
_EXP_DATATYPES = ["str"] + ["number"] * len(EXPERIENCE_KEYS)
_LABEL_TO_KEY = {EXPERIENCE_LABELS_ZH[k]: k for k in EXPERIENCE_KEYS}


def _empty_experience_df(n_rows: int = 0) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        row = {_EXP_NAME_COL: f"参与者{i + 1}"}
        for k in EXPERIENCE_KEYS:
            row[EXPERIENCE_LABELS_ZH[k]] = 3
        rows.append(row)
    return pd.DataFrame(rows, columns=_EXP_HEADERS)


def _persons_to_df(persons: list[dict] | None) -> pd.DataFrame:
    if not persons:
        return _empty_experience_df(0)
    rows = []
    for p in persons:
        exp = p.get("experience") or {}
        row = {_EXP_NAME_COL: p.get("person_name") or p.get("person_id") or ""}
        for k in EXPERIENCE_KEYS:
            try:
                row[EXPERIENCE_LABELS_ZH[k]] = int(round(float(exp.get(k, 3))))
            except (TypeError, ValueError):
                row[EXPERIENCE_LABELS_ZH[k]] = 3
        rows.append(row)
    return pd.DataFrame(rows, columns=_EXP_HEADERS)


def _df_to_persons(df) -> list[dict]:
    """把 Gradio Dataframe / pandas 转回 persons 列表。"""
    if df is None:
        return []
    if isinstance(df, pd.DataFrame):
        frame = df.copy()
    else:
        frame = pd.DataFrame(df)
    if frame.empty:
        return []

    # 兼容列名可能是英文 key 或中文标签
    col_map = {}
    for c in frame.columns:
        c_str = str(c).strip()
        if c_str in (_EXP_NAME_COL, "person_name", "name"):
            col_map[c] = _EXP_NAME_COL
        elif c_str in _LABEL_TO_KEY:
            col_map[c] = EXPERIENCE_LABELS_ZH[_LABEL_TO_KEY[c_str]]
        elif c_str in EXPERIENCE_KEYS:
            col_map[c] = EXPERIENCE_LABELS_ZH[c_str]
        else:
            col_map[c] = c_str
    frame = frame.rename(columns=col_map)

    missing_columns = [
        EXPERIENCE_LABELS_ZH[key]
        for key in EXPERIENCE_KEYS
        if EXPERIENCE_LABELS_ZH[key] not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"体验表缺少列: {', '.join(missing_columns)}")

    persons: list[dict] = []
    for i, row in frame.iterrows():
        name = str(row.get(_EXP_NAME_COL, "") or "").strip()
        score_values = [row.get(EXPERIENCE_LABELS_ZH[key]) for key in EXPERIENCE_KEYS]
        has_scores = any(not pd.isna(value) and value != "" for value in score_values)
        if not name or name.lower() in {"nan", "none"}:
            if not has_scores:
                continue
            name = f"参与者{len(persons) + 1}"

        experience = {}
        for k in EXPERIENCE_KEYS:
            label = EXPERIENCE_LABELS_ZH[k]
            raw = row.get(label)
            if pd.isna(raw) or raw == "":
                raise ValueError(f"{name} 的{label}不能为空")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{name} 的{label}必须是数值") from None
            if not 1.0 <= val <= 5.0:
                raise ValueError(f"{name} 的{label}必须位于1到5之间")
            experience[k] = val

        persons.append(
            {
                "person_id": f"p{len(persons) + 1}",
                "person_name": name,
                "experience": experience,
            }
        )
    return persons


def _suggest_post_edit_df(persons: list[dict] | None) -> pd.DataFrame:
    """改造后体感表初值：沿用同一批人，正向指标略抬高、干扰感略降低。"""
    if not persons:
        return _empty_experience_df(1)
    suggested = []
    for p in persons:
        exp = dict(p.get("experience") or {})
        new_exp = {}
        for k in EXPERIENCE_KEYS:
            v = float(exp.get(k, 3))
            if k == "environmental_disturbance":
                new_exp[k] = max(1.0, min(5.0, round(v - 1)))
            else:
                new_exp[k] = max(1.0, min(5.0, round(min(5.0, v + 1))))
        suggested.append(
            {
                "person_id": p.get("person_id", ""),
                "person_name": p.get("person_name", ""),
                "experience": new_exp,
            }
        )
    return _persons_to_df(suggested)


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


def refresh_image_choices():
    choices = list_scene_choices()
    return gr.update(choices=choices, value=(choices[0] if choices else None))


def on_select_image(image_name: str):
    """下拉选图 → 读 Excel + 分析图，填充前端。"""
    empty_exp = [3.0] * len(EXPERIENCE_KEYS)
    empty_morph = _metrics_to_sliders({k: 0 for k in MORPH_KEYS})
    empty_df = _empty_experience_df(0)
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
        empty_df,
        empty_df,
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
        f"下方表格为九人体感基线（1–5 分，可微调后再点①确认）。"
    )

    pre_df = _persons_to_df(persons)
    post_df = _suggest_post_edit_df(persons)

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
        pre_df,
        post_df,
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
    pre_edit_df,
):
    if not image_name:
        raise gr.Error("请先在下拉列表中选择一张图片")

    try:
        bundle = load_scene_bundle(image_name)
        SESSION["bundle"] = bundle
    except Exception as e:
        raise gr.Error(f"加载场景失败: {e}") from e

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
        pre_edit = _df_to_persons(pre_edit_df)
    except Exception as e:
        raise gr.Error(f"修改前体验表格无效: {e}") from e

    if not pre_edit:
        pre_edit = bundle.get("persons") or []

    images = bundle.get("images") or {}
    original = images.get("original")
    if not original:
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
        else "尚未提供逐人评分；翻译前请在表格中至少填写一名参与者。"
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
    # 改造后体感表：按修改前人员预填建议分，便于对照填写
    post_df = _suggest_post_edit_df(state.get("pre_edit_experience") or bundle.get("persons"))
    return out_path, original, report, post_df, *measured_sliders


def step_save_memory(score, notes, post_edit_df, *corrected_sliders):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("无会话")
    try:
        post_edit = _df_to_persons(post_edit_df)
    except Exception as e:
        raise gr.Error(f"修改后体验表格无效: {e}") from e

    if not post_edit:
        raise gr.Error("请在「改造后多人体验」表格中至少填写一名参与者的评分")

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


def _make_experience_dataframe(label: str, interactive: bool) -> gr.Dataframe:
    return gr.Dataframe(
        value=_empty_experience_df(0),
        headers=_EXP_HEADERS,
        datatype=_EXP_DATATYPES,
        label=label,
        interactive=interactive,
        wrap=True,
        row_count=(9, "dynamic"),
        column_count=(len(_EXP_HEADERS), "fixed"),
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
                pre_edit_df = _make_experience_dataframe(
                    "修改前多人体验（表格，自动从 Excel 九人填入；可微调）",
                    interactive=True,
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

        gr.Markdown("### 改造前后对比")
        with gr.Row():
            out_img = gr.Image(label="修改后全景（TargetIMG）", type="filepath", height=360)
            compare_original_img = gr.Image(
                label="原图对照（改造前）", type="filepath", height=360
            )
        quality_md = gr.Markdown("质检报告将显示在这里")

        gr.Markdown("### 改造后多人体验（表格填写，1–5 分）+ 指标纠偏")
        post_edit_df = _make_experience_dataframe(
            "改造后多人体验（可直接改单元格；生成后会按原班人马预填建议分）",
            interactive=True,
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
            pre_edit_df,
            post_edit_df,
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
                pre_edit_df,
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
            outputs=[out_img, compare_original_img, quality_md, post_edit_df, *corrected_sliders],
        )
        btn_mem.click(
            step_save_memory,
            inputs=[score, notes, post_edit_df, *corrected_sliders],
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
