"""
================================================================================
Gradio 前端（人机协同主界面）v2
文件: app/gradio_app.py
--------------------------------------------------------------------------------
【流程】
  Step0 下拉选择 assets 图片 → 读 filled_metrics.xlsx + 三张分析图
  Step1 表格展示九人体感基线 + 情景/形态
  Step2 调节七个体验目标滑块 → 翻译官
  Step3 人工干预形态要素 → 规划员
  Step4 人工润色自然语言方案 → Seedream 文生图 + 原图对比
  Step5 表格填写修改后多人体验 → 知识库

【启动】
  python app/gradio_app.py
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prefer Chinese-capable fonts when a Windows desktop font is available.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    ASSETS_DIR,
    EXPERIENCE_DIRECTIONS,
    EXPERIENCE_KEYS,
    EXPERIENCE_LABELS_ZH,
    FILLED_METRICS_XLSX,
    MORPH_KEYS,
    MORPH_LABELS_ZH,
    POST_EDIT_METRICS_ENABLED,
)
from app.post_edit_metrics import extract_post_edit_metrics
from pipeline.orchestrator import PipelineOrchestrator
from utils.scene_data import list_scene_choices, load_scene_bundle

pipe = PipelineOrchestrator(
    force_metrics_fallback=True,
    post_edit_metrics_extractor=(
        extract_post_edit_metrics if POST_EDIT_METRICS_ENABLED else None
    ),
)
SESSION: dict = {"state": None, "bundle": None}

# Keep the agent narration in the page itself.  ``gr.Progress`` renders a
# percentage bar below every output component, but this pipeline has no
# meaningful global percentage to expose.
TRANSLATOR_LOADING_STEPS = (
    "正在调用翻译智能体",
    "翻译智能体正在调用知识库",
    "翻译智能体正在分析多参与者体感目标",
    "翻译智能体正在生成七项形态要素目标",
)

CARTOGRAPHER_LOADING_STEPS = (
    "正在调用规划员智能体",
    "规划员智能体正在调用知识库",
    "规划员智能体正在分析场景与空间约束",
    "规划员智能体正在生成最终空间布局方案",
)

GENERATION_LOADING_STEPS = (
    "正在确认最终空间布局方案",
    "正在调用生图智能体进行全景编辑",
    "生图智能体正在生成改造后全景图",
    "正在提取改造后的七项形态要素",
)

AGENT_LOADER_CSS = """
#agent-loader .agent-loader-backdrop {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: rgba(255, 255, 255, 0.24);
    cursor: wait;
}
#agent-loader .agent-loader-card {
    display: flex;
    align-items: center;
    gap: 12px;
    max-width: 620px;
    padding: 12px 16px;
    border: 1px solid #fdba74;
    border-radius: 12px;
    background: #fffaf5;
    box-shadow: 0 12px 32px rgba(154, 52, 18, 0.18);
    color: #9a3412;
    font-size: 16px;
    font-weight: 600;
}
#agent-loader .agent-loader-spinner {
    width: 22px;
    height: 22px;
    flex: 0 0 22px;
    border: 3px solid #fdba74;
    border-top-color: #ea580c;
    border-radius: 50%;
    animation: ai4city-agent-spin 0.8s linear infinite;
}
#agent-loader .agent-loader-elapsed {
    margin-left: 2px;
    color: #c2410c;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px;
    font-variant-numeric: tabular-nums;
}
@keyframes ai4city-agent-spin { to { transform: rotate(360deg); } }
"""


def _show_agent_loader(message: str, elapsed_seconds: int | None = None):
    elapsed = ""
    if elapsed_seconds is not None:
        minutes, seconds = divmod(max(0, elapsed_seconds), 60)
        elapsed = f'<span class="agent-loader-elapsed">已耗时 {minutes:02d}:{seconds:02d}</span>'
    return gr.update(
        value=(
            '<div class="agent-loader-backdrop">'
            '<div class="agent-loader-card" role="status" aria-live="polite">'
            '<span class="agent-loader-spinner" aria-hidden="true"></span>'
            f"<span>{escape(message)}</span>{elapsed}</div>"
            "</div>"
        ),
        visible=True,
    )


def _hide_agent_loader():
    return gr.update(value="", visible=False)


def _unchanged_outputs(count: int) -> tuple:
    return tuple(gr.update() for _ in range(count))


def _run_with_agent_loader(output_count: int, loading_steps: tuple[str, ...], work):
    """Run work immediately while rotating presentation states every five seconds."""
    if len(loading_steps) < 2:
        raise ValueError("加载状态至少需要两个阶段")
    started_at = time.monotonic()

    def loading_update(message: str):
        elapsed = int(time.monotonic() - started_at)
        return (*_unchanged_outputs(output_count), _show_agent_loader(message, elapsed))

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai4city-agent") as executor:
            future = executor.submit(work)
            # The work begins immediately.  The text changes every five seconds
            # only as a presentation cue; it must never delay the real request.
            while not future.done():
                stage_index = min(int((time.monotonic() - started_at) // 5), len(loading_steps) - 1)
                yield loading_update(loading_steps[stage_index])
                time.sleep(1)
            return future.result()
    except Exception:
        yield (*_unchanged_outputs(output_count), _hide_agent_loader())
        raise

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


_DIFF_HEADERS = [
    "体验指标",
    "改造前均值",
    "目标值",
    "改造后均值",
    "改造后-改造前",
    "改造后-目标",
    "目标达成",
]
_ACHIEVE_EPS = 0.05


def _empty_experience_diff_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_DIFF_HEADERS)


def _average_experience(persons: list[dict] | None) -> dict[str, float]:
    """Only aggregate displayed summaries; person-level records remain unchanged."""
    if not persons:
        return {key: 3.0 for key in EXPERIENCE_KEYS}
    totals = {key: 0.0 for key in EXPERIENCE_KEYS}
    for person in persons:
        experience = person.get("experience") or {}
        for key in EXPERIENCE_KEYS:
            totals[key] += float(experience.get(key, 3.0))
    return {key: round(totals[key] / len(persons), 2) for key in EXPERIENCE_KEYS}


def _achievement_label(target: float, post: float, direction: str) -> str:
    if abs(post - target) <= _ACHIEVE_EPS:
        return "已达成"
    if direction == "lower_is_better":
        return "超额达成" if post < target else "未达成"
    return "超额达成" if post > target else "未达成"


def _build_experience_diff_df(
    baseline: dict | None,
    targets: dict | None,
    post_average: dict | None,
) -> pd.DataFrame:
    baseline, targets, post_average = baseline or {}, targets or {}, post_average or {}
    rows = []
    for key in EXPERIENCE_KEYS:
        before = float(baseline.get(key, 3.0))
        target = float(targets.get(key, 3.0))
        after = float(post_average.get(key, 3.0))
        rows.append(
            {
                "体验指标": EXPERIENCE_LABELS_ZH[key],
                "改造前均值": round(before, 2),
                "目标值": round(target, 2),
                "改造后均值": round(after, 2),
                "改造后-改造前": round(after - before, 2),
                "改造后-目标": round(after - target, 2),
                "目标达成": _achievement_label(
                    target,
                    after,
                    EXPERIENCE_DIRECTIONS.get(key, "higher_is_better"),
                ),
            }
        )
    return pd.DataFrame(rows, columns=_DIFF_HEADERS)


def _fmt_experience_diff_summary(diff_df: pd.DataFrame) -> str:
    if diff_df is None or diff_df.empty:
        return "填写改造后多人体验后，将在此汇总改造前、目标与改造后的七项体验。"
    status = diff_df["目标达成"].value_counts().to_dict()
    return (
        "**七项体验目标达成情况** · "
        f"已达成 `{int(status.get('已达成', 0))}` / "
        f"超额达成 `{int(status.get('超额达成', 0))}` / "
        f"未达成 `{int(status.get('未达成', 0))}`\n\n"
        "说明：表中差值均按原始评分相减；环境干扰感为反向指标，数值更低代表更好，"
        "其目标达成判断已按反向规则计算。"
    )


def _build_experience_radar(
    baseline: dict | None,
    targets: dict | None,
    post_average: dict | None,
):
    """Plot three summary profiles; it never replaces the source records."""
    baseline, targets, post_average = baseline or {}, targets or {}, post_average or {}
    labels = [EXPERIENCE_LABELS_ZH[key] for key in EXPERIENCE_KEYS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    def series(values: dict) -> list[float]:
        points = [float(values.get(key, 3.0)) for key in EXPERIENCE_KEYS]
        return points + points[:1]

    figure, axis = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    for name, values, color, linestyle in (
        ("改造前均值", series(baseline), "#4C78A8", "--"),
        ("目标值", series(targets), "#F58518", "-."),
        ("改造后均值", series(post_average), "#54A24B", "-"),
    ):
        axis.plot(closed_angles, values, color=color, linestyle=linestyle, linewidth=2, label=name)
        axis.fill(closed_angles, values, color=color, alpha=0.10)
    axis.set_xticks(angles)
    axis.set_xticklabels(labels, fontsize=10)
    axis.set_ylim(1, 5)
    axis.set_yticks([1, 2, 3, 4, 5])
    axis.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#666666")
    axis.set_title("七项体验对比雷达图", fontsize=13, pad=16)
    axis.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=9, frameon=False)
    figure.tight_layout()
    return figure


def refresh_experience_diff(post_edit_df, *experience_values):
    """Refresh only the derived comparison view, preserving person-level data."""
    state = SESSION.get("state") or {}
    bundle = SESSION.get("bundle") or {}
    baseline = state.get("experience_baseline") or _average_experience(
        state.get("pre_edit_experience") or bundle.get("persons") or []
    )
    targets = state.get("experience_targets") or {
        key: float(value) for key, value in zip(EXPERIENCE_KEYS, experience_values)
    }
    if not targets:
        targets = {key: 3.0 for key in EXPERIENCE_KEYS}
    try:
        post_people = _df_to_persons(post_edit_df)
    except ValueError:
        post_people = []
    post_average = _average_experience(post_people) if post_people else {key: 3.0 for key in EXPERIENCE_KEYS}
    diff_df = _build_experience_diff_df(baseline, targets, post_average)
    return diff_df, _fmt_experience_diff_summary(diff_df), _build_experience_radar(
        baseline, targets, post_average
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


def refresh_image_choices():
    choices = list_scene_choices()
    return gr.update(choices=choices, value=(choices[0] if choices else None))


def on_select_image(
    image_name: str,
):
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
        yield (*blank, _hide_agent_loader())
        return

    # Keep a short, visible loading beat while the four derived scene images and
    # Excel-backed bundle are prepared for the interface.
    yield (*_unchanged_outputs(len(blank)), _show_agent_loader("正在进行数据处理"))
    time.sleep(0.5)
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

    yield (
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
        _hide_agent_loader(),
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
        state = yield from _run_with_agent_loader(
            1 + len(MORPH_KEYS),
            TRANSLATOR_LOADING_STEPS,
            lambda: pipe.run_translator(state, targets),
        )
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
        f"学习Agent: {'已参考' if tr.get('learning_applied') else '启用'}"
    )
    sliders = _metrics_to_sliders(state["confirmed_target_metrics"])
    yield info, *sliders, _hide_agent_loader()


def _fmt_layout_plan(plan: dict) -> str:
    actions = []
    for item in plan.get("object_actions") or []:
        actions.append(
            f"- **{item.get('action', 'adjust')} {item.get('object_type', '空间对象')}**："
            f"{item.get('position', '')}；{item.get('quantity', '')}"
        )
    return (
        "### 审查依据（不单独发送给 Seedream）\n"
        "最终执行内容以左侧/上方可编辑的「最终执行空间布局方案」为准；"
        "点击生成后，该文本将原样发送。\n\n"
        f"### 原始专家意见（已供规划员参考）\n{plan.get('expert_advice') or '—'}\n\n"
        f"### 方案摘要\n{plan.get('plan_summary', '')}\n\n"
        f"### 完整对象级修改（审查用）\n{chr(10).join(actions) or '- 轻量优化现有空间对象'}\n\n"
        f"### 完整保持不变区域（审查用）\n"
        + "\n".join(f"- {x}" for x in plan.get("unchanged_regions") or [])
        + f"\n\n### 生成依据\n{plan.get('rationale', '')}"
    )


def step_confirm_morph(
    expert_advice,
    *slider_vals,
):
    state = SESSION.get("state")
    if not state:
        raise gr.Error("请先完成「确认体验滑块 → 翻译官」")
    human_metrics = _sliders_to_metrics(list(slider_vals))
    note = (expert_advice or "").strip() or "前端人工确认形态要素"
    state = yield from _run_with_agent_loader(
        2,
        CARTOGRAPHER_LOADING_STEPS,
        lambda: pipe.confirm_morph(
            state,
            human_metrics=human_metrics,
            note=note,
            language="zh",
        ),
    )
    SESSION["state"] = state
    plan = state.get("modification_plan") or {}
    review_path = state.get("review_record_path") or ""
    review_notice = (
        f"\n\n### Markdown 复核记录\n`{review_path}`"
        if review_path
        else ""
    )
    yield plan.get("draft_text", ""), _fmt_layout_plan(plan) + review_notice, _hide_agent_loader()


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

    output_count = 3 + len(MORPH_KEYS) + 3
    state = yield from _run_with_agent_loader(
        output_count,
        GENERATION_LOADING_STEPS,
        lambda: pipe.generate_and_check(pipe.confirm_plan(state, final_plan)),
    )
    SESSION["state"] = state

    gen = state["generation"]
    qr = state.get("quality_report") or {}
    out_path = gen.get("output_image_path")
    if out_path:
        out_path = str(Path(out_path).resolve())
        if not Path(out_path).is_file():
            raise gr.Error(f"生成文件不存在，无法展示: {out_path}")

    # Task 4 branch keeps the generation UI concise: use measured values only
    # to prefill correction sliders, without a separate textual quality report.
    measured = qr.get("measured_metrics") or {}
    measured_sliders = (
        _metrics_to_sliders(measured)
        if measured
        else [gr.update() for _ in MORPH_KEYS]
    )
    # 改造后体感表：按修改前人员预填建议分，便于对照填写
    post_df = _suggest_post_edit_df(state.get("pre_edit_experience") or bundle.get("persons"))
    diff_df, diff_md, radar = refresh_experience_diff(post_df)
    yield (
        out_path,
        original,
        post_df,
        *measured_sliders,
        diff_df,
        diff_md,
        radar,
        _hide_agent_loader(),
    )


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
    diff_df, diff_md, radar = refresh_experience_diff(post_edit_df)
    return (
        f"已写入知识库，memory_id = {state['memory_id']}，session = {state['session_id']}\n"
        f"学习Agent统计: {json.dumps(learning_stats, ensure_ascii=False)}",
        diff_df,
        diff_md,
        radar,
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
            # 高密度情境下微空间优化 · 人机多智能体流水线
            **选图(Excel指标+分析图) → 体验滑块 → 翻译官 → 规划员 → Seedream 文生图 → 反馈入库**
            """
        )
        agent_loader = gr.HTML(value="", visible=False, elem_id="agent-loader")

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
            label="专家意见（可选，供规划员转写为最终方案）",
            placeholder="例如：保持左侧历史建筑立面不变，优先优化右侧停留空间",
            lines=2,
        )
        btn_morph = gr.Button(
            "③ 确认形态要素 → 规划员生成空间布局方案", variant="primary"
        )

        plan_box = gr.Textbox(
            label="人工干预②：最终执行空间布局方案（可直接修改；确认后原样发送给 Seedream）",
            lines=8,
        )
        plan_rationale = gr.Markdown("")
        btn_gen = gr.Button("④ 确认方案 → Seedream 文生图", variant="primary")

        gr.Markdown("### 改造前后对比")
        with gr.Row():
            out_img = gr.Image(label="修改后全景（TargetIMG）", type="filepath", height=360)
            compare_original_img = gr.Image(
                label="原图对照（改造前）", type="filepath", height=360
            )

        gr.Markdown("### 改造后多人体验（表格填写，1–5 分）+ 指标纠偏")
        post_edit_df = _make_experience_dataframe(
            "改造后多人体验（可直接改单元格；生成后会按原班人马预填建议分）",
            interactive=True,
        )
        gr.Markdown("### 体验前后差异（改造前 / 目标 / 改造后）")
        experience_diff_df = gr.Dataframe(
            value=_empty_experience_diff_df(),
            headers=_DIFF_HEADERS,
            datatype=["str"] + ["number"] * 5 + ["str"],
            label="七项体验汇总（逐人评分保留在上方表格）",
            interactive=False,
            wrap=True,
            row_count=(len(EXPERIENCE_KEYS), "fixed"),
            column_count=(len(_DIFF_HEADERS), "fixed"),
        )
        experience_diff_md = gr.Markdown(
            "填写改造后多人体验后，将在此汇总改造前、目标与改造后的七项体验。"
        )
        experience_radar = gr.Plot(
            label="七项体验对比雷达图（蓝=改造前 / 橙=目标 / 绿=改造后）"
        )
        btn_refresh_diff = gr.Button("刷新体验前后差异")
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
        image_dropdown.change(
            on_select_image,
            inputs=[image_dropdown],
            outputs=[*select_outputs, agent_loader],
            show_progress="hidden",
        )
        demo.load(
            on_select_image,
            inputs=[image_dropdown],
            outputs=[*select_outputs, agent_loader],
            show_progress="hidden",
        )

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
            outputs=[translate_md, *morph_sliders, agent_loader],
            show_progress="hidden",
        )
        btn_morph.click(
            step_confirm_morph,
            inputs=[expert_advice, *morph_sliders],
            outputs=[plan_box, plan_rationale, agent_loader],
            show_progress="hidden",
        )
        btn_gen.click(
            step_generate,
            inputs=[plan_box],
            outputs=[
                out_img,
                compare_original_img,
                post_edit_df,
                *corrected_sliders,
                experience_diff_df,
                experience_diff_md,
                experience_radar,
                agent_loader,
            ],
            show_progress="hidden",
        )
        post_edit_df.change(
            refresh_experience_diff,
            inputs=[post_edit_df, *experience_sliders],
            outputs=[experience_diff_df, experience_diff_md, experience_radar],
        )
        btn_refresh_diff.click(
            refresh_experience_diff,
            inputs=[post_edit_df, *experience_sliders],
            outputs=[experience_diff_df, experience_diff_md, experience_radar],
        )
        btn_mem.click(
            step_save_memory,
            inputs=[score, notes, post_edit_df, *corrected_sliders],
            outputs=[mem_out, experience_diff_df, experience_diff_md, experience_radar],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        css=AGENT_LOADER_CSS,
    )

