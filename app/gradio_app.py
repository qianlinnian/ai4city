"""AI4City 四阶段 Gradio 工作台。

在线流程只消费 Task 1 已写入大表的七项形态指标；本模块不会导入或调用
任何图像指标提取器。浏览器态保存在 ``gr.State``，阶段完成后再写入会话文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.data_catalog import scan_backend_catalog
from app.generation_backend import default_backend, generate
from app.metrics_table_loader import (
    MetricsTable,
    MetricsTableError,
    compare_metrics,
    list_workbook_sheets,
)
from app.session_store import (
    COMPLETED,
    GENERATED,
    INPUT_PENDING,
    MORPH_REVIEW,
    PLAN_CONFIRMED,
    PLAN_REVIEW,
    VALIDATION_PENDING,
    SessionStore,
)
from app.ui_theme import APP_CSS, APP_THEME, stage_stepper_html, status_html
from config import (
    EXPERIENCE_KEYS,
    EXPERIENCE_LABELS_ZH,
    MORPH_KEYS,
    MORPH_LABELS_ZH,
    SESSION_DIR,
    UPLOAD_DIR,
)
from pipeline.orchestrator import PipelineOrchestrator


PIPELINE = PipelineOrchestrator()
SESSION_STORE = SessionStore(SESSION_DIR, UPLOAD_DIR)

EXPERIENCE_HEADERS = [
    "参与者 ID",
    "姓名",
    *[EXPERIENCE_LABELS_ZH[key] for key in EXPERIENCE_KEYS],
]
DEFAULT_PARTICIPANTS = [["P01", "", 3, 3, 3, 3, 3, 3, 3]]
MORPH_DISPLAY_HEADERS = ["指标", "Excel 原始值", "Agent 目标值", "变化方向"]
MORPH_REVIEW_HEADERS = [*MORPH_DISPLAY_HEADERS, "人工确认值"]
STAGE_TO_STEP = {
    INPUT_PENDING: 0,
    MORPH_REVIEW: 1,
    PLAN_REVIEW: 2,
    PLAN_CONFIRMED: 2,
    GENERATED: 3,
    VALIDATION_PENDING: 3,
    COMPLETED: 3,
}


def _as_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        return value.values.tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [list(row) for row in value]
    raise ValueError("表格数据格式无法识别，请重新填写")


def parse_participants(value: Any, *, required: bool = True) -> list[dict[str, Any]]:
    """把可编辑表格转换为逐人七项记录；绝不在前端求平均。"""

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(_as_rows(value), start=1):
        padded = [*row, *([None] * max(0, len(EXPERIENCE_HEADERS) - len(row)))]
        person_id = str(padded[0] or "").strip()
        name = str(padded[1] or "").strip()
        score_cells = padded[2 : 2 + len(EXPERIENCE_KEYS)]
        if not person_id and not name and all(cell in (None, "") for cell in score_cells):
            continue
        if not person_id:
            raise ValueError(f"第 {row_number} 位参与者缺少 ID")
        if person_id in seen_ids:
            raise ValueError(f"参与者 ID 重复：{person_id}")
        scores: dict[str, float] = {}
        for key, cell in zip(EXPERIENCE_KEYS, score_cells):
            try:
                score = float(cell)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"参与者 {person_id} 的{EXPERIENCE_LABELS_ZH[key]}必须填写 1～5 分"
                ) from exc
            if not 1 <= score <= 5:
                raise ValueError(
                    f"参与者 {person_id} 的{EXPERIENCE_LABELS_ZH[key]}超出 1～5 分"
                )
            scores[key] = score
        records.append(
            {"person_id": person_id, "person_name": name, "experience": scores}
        )
        seen_ids.add(person_id)
    if required and not records:
        raise ValueError("请至少填写一名参与者的完整七项体验评分")
    return records


def participants_to_rows(records: Iterable[dict[str, Any]] | None) -> list[list[Any]]:
    rows = []
    for item in records or []:
        exp = item.get("experience") or {}
        rows.append(
            [
                item.get("person_id", ""),
                item.get("person_name", ""),
                *[exp.get(key, 3) for key in EXPERIENCE_KEYS],
            ]
        )
    return rows or [list(DEFAULT_PARTICIPANTS[0])]


def _metric_display(key: str, value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.2f}" if key == "color_richness" else f"{number * 100:.2f}%"


def _metric_slider_value(key: str, value: Any) -> float:
    number = float(value)
    return number if key == "color_richness" else number * 100


def sliders_to_metrics(values: Iterable[Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, raw in zip(MORPH_KEYS, values):
        number = float(raw)
        metrics[key] = number if key == "color_richness" else number / 100
    return metrics


def morph_summary_rows(
    baseline: dict[str, Any] | None,
    agent: dict[str, Any] | None = None,
    human: dict[str, Any] | None = None,
) -> list[list[Any]]:
    baseline = baseline or {}
    agent = agent or baseline
    rows: list[list[Any]] = []
    for key in MORPH_KEYS:
        before = baseline.get(key)
        target = agent.get(key)
        direction = "—"
        if before is not None and target is not None:
            delta = float(target) - float(before)
            direction = "↑ 增加" if delta > 1e-9 else "↓ 降低" if delta < -1e-9 else "— 保持"
        row = [
            MORPH_LABELS_ZH[key],
            _metric_display(key, before),
            _metric_display(key, target),
            direction,
        ]
        if human is not None:
            row.append(_metric_display(key, human.get(key)))
        rows.append(row)
    return rows


def _candidate_choices(candidates: Iterable[Any]) -> list[str]:
    return [
        f"第 {item.row_index} 行 · {item.image_name} · 相似度 {item.score:.0%}"
        for item in candidates
    ]


def _candidate_row_index(choice: str | None) -> int | None:
    if not choice:
        return None
    try:
        return int(str(choice).split("行", 1)[0].replace("第", "").strip())
    except (TypeError, ValueError):
        return None


def _resolve_metrics(
    image_path: str | None,
    table_path: str | None,
    sheet_name: str | None,
    candidate_choice: str | None,
) -> tuple[dict[str, float], int, str]:
    if not image_path:
        raise ValueError("请先上传原始全景图")
    if not table_path:
        raise ValueError("请上传包含 Task 1 结果的项目大表（XLSX/XLSM/CSV）")
    table = MetricsTable.from_file(table_path, sheet_name=sheet_name or None)
    selected_row = _candidate_row_index(candidate_choice)
    if selected_row is not None:
        return table.metrics_for_row(selected_row), selected_row, "人工选择候选行"
    match = table.match_image(Path(image_path).name)
    if match.matched and match.metrics is not None and match.row_index is not None:
        return match.metrics, match.row_index, f"{match.match_type or '名称'}匹配"
    if match.status == "invalid":
        raise ValueError(match.error or "匹配行的指标为空、格式错误或越界")
    if match.candidates:
        raise ValueError("没有唯一匹配，请从候选行中人工选择后再继续")
    raise ValueError("大表中未找到与图片名称对应的指标行")


def load_sheet_choices(table_file: str | None):
    if not table_file:
        return gr.update(choices=[], value=None), status_html("等待选择后端项目大表")
    try:
        sheets = list_workbook_sheets(table_file)
        return (
            gr.update(choices=sheets, value=sheets[0] if sheets else None),
            status_html(f"已读取 {len(sheets)} 个工作表", "ok"),
        )
    except Exception as exc:
        return gr.update(choices=[], value=None), status_html(str(exc), "error")


def refresh_backend_catalog():
    """重新扫描后端数据目录，并刷新三个数据选择框。"""

    catalog = scan_backend_catalog()
    image_choices = [item.as_choice() for item in catalog.images]
    table_choices = [item.as_choice() for item in catalog.metric_tables]
    image_value = catalog.images[0].path if catalog.images else None
    table_value = catalog.metric_tables[0].path if catalog.metric_tables else None
    if catalog.images and catalog.metric_tables:
        message = (
            f"后端目录已发现 {len(catalog.images)} 张全景图、"
            f"{len(catalog.metric_tables)} 个有效项目大表"
        )
        kind = "ok"
    else:
        missing = []
        if not catalog.images:
            missing.append("全景图")
        if not catalog.metric_tables:
            missing.append("含七项形态指标的项目大表")
        message = "后端数据目录尚未发现" + "和".join(missing)
        kind = "warn"
    return (
        gr.update(choices=image_choices, value=image_value),
        gr.update(choices=table_choices, value=table_value),
        gr.update(choices=table_choices, value=table_value),
        status_html(message, kind),
    )


def select_backend_image(image_path: str | None):
    if not image_path:
        return None, "—"
    path = Path(image_path)
    if not path.is_file():
        raise gr.Error(f"后端图片不存在：{path.name}")
    return str(path), path.name


def preview_baseline(
    image_path: str | None,
    table_path: str | None,
    sheet_name: str | None,
    candidate_choice: str | None,
):
    if not image_path or not table_path:
        return [], gr.update(choices=[], value=None), status_html(
            "请选择后端全景图和项目大表，系统将按图片文件名匹配", "warn"
        ), Path(image_path).name if image_path else "—"
    try:
        table = MetricsTable.from_file(table_path, sheet_name=sheet_name or None)
        selected = _candidate_row_index(candidate_choice)
        if selected is not None:
            metrics = table.metrics_for_row(selected)
            return (
                morph_summary_rows(metrics),
                gr.update(),
                status_html(f"已采用人工选择的第 {selected} 行", "ok"),
                Path(image_path).name,
            )
        match = table.match_image(Path(image_path).name)
        choices = _candidate_choices(match.candidates)
        if match.matched:
            return (
                morph_summary_rows(match.metrics),
                gr.update(choices=choices, value=None),
                status_html(
                    f"已匹配：第 {match.row_index} 行 · {match.image_name}", "ok"
                ),
                Path(image_path).name,
            )
        message = match.error or (
            "存在多个同名/同 stem 行，请人工选择"
            if match.status == "ambiguous"
            else "未精确匹配，请核对图片名称或人工选择候选行"
        )
        return (
            [],
            gr.update(choices=choices, value=None),
            status_html(message, "error" if match.status == "invalid" else "warn"),
            Path(image_path).name,
        )
    except Exception as exc:
        return [], gr.update(choices=[], value=None), status_html(str(exc), "error"), Path(image_path).name


def _scene_context(
    space_type: str,
    current_use: str,
    main_people: str,
    time_weather: str,
    description: str,
) -> dict[str, str]:
    return {
        "space_type": str(space_type or "").strip(),
        "observation_time": str(time_weather or "").strip(),
        "description": "；".join(
            item
            for item in [
                f"当前用途：{str(current_use).strip()}" if current_use else "",
                f"主要人群：{str(main_people).strip()}" if main_people else "",
                str(description or "").strip(),
            ]
            if item
        ),
    }


def confirm_inputs(
    image_path: str | None,
    table_path: str | None,
    sheet_name: str | None,
    candidate_choice: str | None,
    space_type: str,
    current_use: str,
    main_people: str,
    time_weather: str,
    description: str,
    participants: Any,
    *target_values: Any,
):
    try:
        metrics, row_index, match_type = _resolve_metrics(
            image_path, table_path, sheet_name, candidate_choice
        )
        records = parse_participants(participants)
        targets = {key: float(value) for key, value in zip(EXPERIENCE_KEYS, target_values)}
        context = _scene_context(space_type, current_use, main_people, time_weather, description)
        state = PIPELINE.start_session(
            image_path, metrics, scene_context=context, pre_edit_experience=records
        )
        state.update(
            {
                "image_path": str(Path(image_path).resolve()),
                "metrics_table_path": str(Path(table_path).resolve()),
                "metrics_sheet": sheet_name or "",
                "metrics_row_index": row_index,
                "metrics_match_type": match_type,
                "data_source": "backend_catalog",
            }
        )
        state = PIPELINE.run_translator(state, targets, experience_records=records)
        state = SESSION_STORE.save_session(state)
        translation = state["morph_translation"]
        agent = translation["target_metrics"]
        evidence = "\n".join(
            f"- {item.get('method', 'unknown').upper()}：{item.get('summary', '')}"
            for item in translation.get("conversion_basis") or []
        ) or "- 暂无额外证据"
        scene = state.get("scene_understanding") or {}
        scene_status = scene.get("status", "not_run")
        view_count = len(state.get("panorama_views") or [])
        evidence += f"\n- SCENE：{scene_status}，共 {view_count} 张派生视图"
        if scene.get("degradation_reason"):
            evidence += f"；{scene['degradation_reason']}"
        warnings = (state.get("task2_reasonableness") or {}).get("warnings") or []
        evidence += "".join(f"\n- CHECK：{item}" for item in warnings)
        return (
            state,
            stage_stepper_html(1),
            status_html(f"翻译官已完成 · 会话 {state['session_id']}", "ok"),
            state["session_id"],
            state["image_path"],
            state.get("scene_context_text", ""),
            str(len(records)),
            json.dumps(targets, ensure_ascii=False, indent=2),
            morph_summary_rows(metrics, agent),
            translation.get("rationale", ""),
            evidence,
            *[_metric_slider_value(key, agent[key]) for key in MORPH_KEYS],
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def reset_morph_sliders(state: dict[str, Any], source: str):
    if not state or state.get("stage") != MORPH_REVIEW:
        raise gr.Error("当前没有待审核的形态目标")
    if source == "baseline":
        metrics = state["baseline_metrics"]
    else:
        metrics = state["morph_translation"]["target_metrics"]
    return [_metric_slider_value(key, metrics[key]) for key in MORPH_KEYS]


def _plan_action_rows(plan: dict[str, Any]) -> list[list[str]]:
    labels = {"add": "新增", "remove": "删除", "adjust": "调整"}
    return [
        [
            labels.get(item.get("action"), item.get("action", "")),
            item.get("object_type", ""),
            item.get("position", ""),
            item.get("quantity", ""),
            "、".join(item.get("attributes") or []),
            item.get("rationale", ""),
        ]
        for item in plan.get("object_actions") or []
    ]


def confirm_morph(state: dict[str, Any], expert_note: str, *slider_values: Any):
    try:
        if not state:
            raise ValueError("请先完成输入与体验阶段")
        metrics = sliders_to_metrics(slider_values)
        state = PIPELINE.confirm_morph(state, metrics, note=expert_note, language="en")
        state = SESSION_STORE.save_session(state)
        plan = state["modification_plan"]
        relations = "\n".join(f"- {item}" for item in plan.get("spatial_relations") or [])
        unchanged = "\n".join(f"- {item}" for item in plan.get("unchanged_regions") or [])
        constraints = "\n".join(f"- {item}" for item in plan.get("constraints") or [])
        return (
            state,
            stage_stepper_html(2),
            status_html("专家形态目标已确认，空间方案已生成", "ok"),
            state["image_path"],
            morph_summary_rows(state["baseline_metrics"], metrics, metrics),
            expert_note or "未填写额外专家建议",
            _plan_action_rows(plan),
            relations or "- 暂无",
            unchanged or "- 暂无",
            constraints or "- 暂无",
            plan.get("draft_text", ""),
            plan.get("draft_text", ""),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def restore_agent_prompt(state: dict[str, Any]) -> str:
    if not state or not state.get("modification_plan"):
        raise gr.Error("当前没有 Agent 原始方案")
    return state["modification_plan"].get("draft_text", "")


def generate_panorama(state: dict[str, Any], prompt: str, backend: str):
    try:
        if not state:
            raise ValueError("请先完成空间布局方案")
        state = PIPELINE.confirm_plan(state, prompt)
        result = generate(state["image_path"], state["final_prompt"], backend=backend)
        state = PIPELINE.record_generation(state, result)
        state = SESSION_STORE.save_session(state)
        raw = result.raw or {}
        selected = str(raw.get("backend") or backend)
        mode = "MOCK" if result.mock else "LIVE"
        fallback = raw.get("fallback_reason") or raw.get("error") or "无"
        detail = f"后端：{selected} · 模式：{mode} · Fallback：{fallback}"
        return (
            state,
            stage_stepper_html(3),
            status_html("全景图已生成，可以进行人工验证", "ok"),
            state["image_path"],
            result.output_image_path,
            detail,
            status_html("尚未提供修改后指标，当前仅进行人工体验验证", "warn"),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def preview_post_metrics(
    state: dict[str, Any],
    table_path: str | None,
    sheet_name: str | None,
    candidate_choice: str | None,
):
    if not state or state.get("stage") not in {GENERATED, VALIDATION_PENDING}:
        raise gr.Error("请先生成全景图")
    if not table_path:
        return [], gr.update(choices=[], value=None), status_html(
            "尚未提供修改后指标，当前仅进行人工体验验证", "warn"
        ), state
    try:
        output_path = (state.get("generation") or {}).get("output_image_path")
        metrics, row_index, _ = _resolve_metrics(
            output_path, table_path, sheet_name, candidate_choice
        )
        state = PIPELINE.record_quality_metrics(state, metrics)
        state["post_metrics_table_path"] = table_path
        state["post_metrics_sheet"] = sheet_name or ""
        state["post_metrics_row_index"] = row_index
        state = SESSION_STORE.save_session(state)
        rows = []
        for item in compare_metrics(
            state["baseline_metrics"], metrics, state["confirmed_target_metrics"]
        ):
            threshold = 1.5 if item.key == "color_richness" else 0.06
            deviation = abs(float(item.target_deviation or 0))
            result = "达到目标" if deviation <= threshold / 2 else "接近目标" if deviation <= threshold else "偏差较大"
            rows.append(
                [
                    item.label,
                    _metric_display(item.key, item.before),
                    _metric_display(item.key, item.target),
                    _metric_display(item.key, item.after),
                    _metric_display(item.key, item.target_deviation),
                    result,
                ]
            )
        return (
            rows,
            gr.update(choices=[], value=None),
            status_html((state.get("quality_report") or {}).get("details", "指标对比完成"), "ok"),
            state,
        )
    except Exception as exc:
        try:
            table = MetricsTable.from_file(table_path, sheet_name=sheet_name or None)
            output_path = (state.get("generation") or {}).get("output_image_path", "")
            choices = _candidate_choices(table.match_image(Path(output_path).name).candidates)
        except Exception:
            choices = []
        return [], gr.update(choices=choices, value=None), status_html(str(exc), "warn"), state


def finish_session(
    state: dict[str, Any], post_participants: Any, expert_score: float, notes: str
):
    try:
        if not state:
            raise ValueError("当前没有可保存的会话")
        records = parse_participants(post_participants, required=False)
        if records:
            state = PIPELINE.record_post_experience(state, records)
        state = PIPELINE.save_memory(state, score=float(expert_score), notes=notes)
        state = SESSION_STORE.save_session(state)
        return (
            state,
            stage_stepper_html(3),
            status_html(f"已保存并完成 · 记忆 ID：{state.get('memory_id', '—')}", "ok"),
            f"会话 {state['session_id']} 已完成，可随时通过会话 ID 恢复。",
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def rollback_session(state: dict[str, Any], target_stage: str):
    try:
        if not state:
            raise ValueError("当前没有会话")
        state = SESSION_STORE.rollback(state, target_stage)
        return (
            state,
            stage_stepper_html(STAGE_TO_STEP[target_stage]),
            status_html("已回退；之前填写的数据仍保留", "ok"),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def recover_session(session_id: str):
    try:
        state = SESSION_STORE.load_session(str(session_id or "").strip())
        stage = state.get("stage", INPUT_PENDING)
        translation = state.get("morph_translation") or {}
        agent = translation.get("target_metrics") or state.get("baseline_metrics") or {}
        plan = state.get("modification_plan") or {}
        generation = state.get("generation") or {}
        return (
            state,
            stage_stepper_html(STAGE_TO_STEP.get(stage, 0)),
            status_html(f"已恢复会话 · 当前阶段：{stage}", "ok"),
            state.get("image_path"),
            state.get("image_path"),
            participants_to_rows(state.get("pre_edit_experience")),
            morph_summary_rows(state.get("baseline_metrics"), agent),
            *[_metric_slider_value(key, agent.get(key, 0)) for key in MORPH_KEYS],
            _plan_action_rows(plan),
            plan.get("draft_text", ""),
            state.get("image_path"),
            generation.get("output_image_path"),
            participants_to_rows(state.get("post_edit_experience")) if state.get("post_edit_experience") else [],
        )
    except Exception as exc:
        raise gr.Error(f"会话恢复失败：{exc}") from exc


def _section_header(kicker: str, title: str) -> None:
    gr.HTML(f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>')


def build_ui() -> gr.Blocks:
    catalog = scan_backend_catalog()
    image_choices = [item.as_choice() for item in catalog.images]
    table_choices = [item.as_choice() for item in catalog.metric_tables]
    initial_image = catalog.images[0].path if catalog.images else None
    initial_table = catalog.metric_tables[0].path if catalog.metric_tables else None
    if catalog.images and catalog.metric_tables:
        catalog_message = status_html(
            f"后端目录已发现 {len(catalog.images)} 张全景图、"
            f"{len(catalog.metric_tables)} 个有效项目大表",
            "ok",
        )
    else:
        catalog_message = status_html(
            "请在后端 data 目录配置全景图和包含七项指标的项目大表",
            "warn",
        )
    with gr.Blocks(
        title="AI4City · 全景空间优化工作台",
    ) as demo:
        browser_state = gr.State({})
        gr.HTML(
            """
            <section class="app-hero">
              <div class="app-eyebrow">AI4CITY · PANORAMA WORKBENCH</div>
              <h1>全景空间优化工作台</h1>
              <p>从项目大表读取形态基线，连接体验目标、形态审核、空间方案与生成验证。</p>
              <div class="hero-tags"><span class="hero-tag">Excel 驱动</span>
              <span class="hero-tag">多人体验保真</span><span class="hero-tag">可回退会话</span></div>
            </section>
            """
        )
        stepper = gr.HTML(stage_stepper_html(0))
        global_status = gr.HTML(status_html("等待开始新会话"))

        with gr.Accordion("恢复已有会话", open=False):
            with gr.Row():
                recovery_id = gr.Textbox(label="会话 ID", placeholder="输入 outputs/sessions 中的 session_id")
                recovery_button = gr.Button("恢复会话", variant="secondary")
            recovered_session_id = gr.Textbox(label="当前会话 ID", interactive=False)

        with gr.Tabs() as main_tabs:
            with gr.Tab("01 · 输入与体验", id=0):
                with gr.Row():
                    with gr.Column(scale=5, min_width=420):
                        _section_header("INPUT", "原始全景与项目大表")
                        with gr.Row():
                            image_selector = gr.Dropdown(
                                choices=image_choices,
                                value=initial_image,
                                label="从后端选择全景图",
                                filterable=True,
                            )
                            refresh_catalog_button = gr.Button("刷新后端数据目录")
                        input_image = gr.Image(
                            value=initial_image,
                            type="filepath", label="原始全景图预览", height=320,
                            interactive=False,
                            elem_classes=["image-stage"],
                        )
                        image_name = gr.Textbox(
                            label="图片文件名",
                            value=Path(initial_image).name if initial_image else "—",
                            interactive=False,
                        )
                        metrics_file = gr.Dropdown(
                            choices=table_choices,
                            value=initial_table,
                            label="从后端选择项目大表",
                            filterable=True,
                        )
                        with gr.Row():
                            metrics_sheet = gr.Dropdown(label="工作表", choices=[])
                            metrics_candidate = gr.Dropdown(label="候选指标行", choices=[])
                        backend_catalog_status = gr.HTML(catalog_message)
                        metrics_status = gr.HTML(
                            status_html("等待选择后端项目大表")
                        )
                    with gr.Column(scale=7, min_width=520):
                        _section_header("CONTEXT", "情景要素与逐人体验")
                        with gr.Row():
                            space_type = gr.Textbox(label="地点 / 场景类型", placeholder="社区街巷、滨水空间…")
                            current_use = gr.Textbox(label="当前用途", placeholder="通行、休憩、活动…")
                        with gr.Row():
                            main_people = gr.Textbox(label="主要人群", placeholder="居民、儿童、访客…")
                            time_weather = gr.Textbox(label="时间或天气", placeholder="午后、晴；傍晚、阴…")
                        context_description = gr.Textbox(label="其他描述", lines=2)
                        gr.Markdown("#### 修改前参与者体验评分（1～5；环境干扰感越低越好）")
                        pre_participants = gr.Dataframe(
                            value=DEFAULT_PARTICIPANTS,
                            headers=EXPERIENCE_HEADERS,
                            datatype=["str", "str", *(["number"] * 7)],
                            row_count=(1, "dynamic"),
                            column_count=(9, "fixed"),
                            interactive=True,
                            wrap=True,
                            label="可添加或删除参与者；每一行完整保留",
                        )
                        with gr.Accordion("七项体验目标", open=True):
                            experience_sliders = [
                                gr.Slider(
                                    1, 5, value=2 if key == "environmental_disturbance" else 4,
                                    step=0.1, label=EXPERIENCE_LABELS_ZH[key],
                                )
                                for key in EXPERIENCE_KEYS
                            ]
                baseline_table = gr.Dataframe(
                    headers=MORPH_DISPLAY_HEADERS, value=[], interactive=False,
                    label="从大表读取的七项形态基线", elem_classes=["metric-grid"],
                )
                confirm_input_button = gr.Button(
                    "确认体验目标并运行翻译官", variant="primary", elem_classes=["primary-action"]
                )

            with gr.Tab("02 · 形态目标审核", id=1):
                with gr.Row():
                    with gr.Column(scale=4):
                        _section_header("REVIEW", "输入摘要")
                        morph_image = gr.Image(type="filepath", label="原始全景图", height=250)
                        morph_context = gr.Textbox(label="情景要素", interactive=False, lines=3)
                        morph_people_count = gr.Textbox(label="参与者数量", interactive=False)
                        morph_targets_json = gr.Code(label="七项体验目标", language="json", interactive=False)
                    with gr.Column(scale=8):
                        _section_header("TRANSLATION", "形态目标对比与人工确认")
                        morph_compare_table = gr.Dataframe(
                            headers=MORPH_DISPLAY_HEADERS, interactive=False, elem_classes=["metric-grid"]
                        )
                        with gr.Row():
                            restore_agent_button = gr.Button("恢复 Agent 建议值")
                            restore_baseline_button = gr.Button("恢复 Excel 原始值")
                        morph_sliders = []
                        for key in MORPH_KEYS:
                            maximum = 24 if key == "color_richness" else 100
                            suffix = "" if key == "color_richness" else "（%）"
                            morph_sliders.append(
                                gr.Slider(0, maximum, value=0, step=0.1, label=f"{MORPH_LABELS_ZH[key]}{suffix}")
                            )
                    with gr.Column(scale=4):
                        _section_header("EVIDENCE", "解释、证据与专家建议")
                        translator_rationale = gr.Textbox(label="翻译说明", interactive=False, lines=4)
                        translator_evidence = gr.Markdown("暂无证据")
                        expert_note = gr.Textbox(label="专家补充建议", lines=6)
                        confirm_morph_button = gr.Button(
                            "专家确认形态目标", variant="primary", elem_classes=["primary-action"]
                        )

            with gr.Tab("03 · 空间布局方案", id=2):
                with gr.Row():
                    plan_image = gr.Image(type="filepath", label="原始全景图", height=245)
                    plan_metrics = gr.Dataframe(
                        headers=MORPH_REVIEW_HEADERS, interactive=False,
                        label="已确认的七项形态目标", elem_classes=["metric-grid"]
                    )
                    plan_expert_note = gr.Textbox(label="专家建议摘要", interactive=False, lines=5)
                _section_header("LAYOUT", "对象级空间动作")
                action_table = gr.Dataframe(
                    headers=["动作", "对象", "位置", "数量 / 尺度", "属性", "目的"],
                    interactive=False, wrap=True, elem_classes=["metric-grid"]
                )
                with gr.Row():
                    plan_relations = gr.Markdown("### 空间关系\n- 暂无")
                    plan_unchanged = gr.Markdown("### 保持不变区域\n- 暂无")
                    plan_constraints = gr.Markdown("### 约束\n- 暂无")
                agent_prompt = gr.Textbox(label="模型原始生成文本", interactive=False, lines=7)
                final_prompt = gr.Textbox(label="专家可编辑的最终生成文本", lines=9)
                with gr.Row():
                    restore_prompt_button = gr.Button("恢复 Agent 原始方案")
                    backend_selector = gr.Dropdown(
                        choices=["mock", "seedream", "worldlabs"],
                        value=default_backend(), label="生成后端",
                    )
                    generate_button = gr.Button(
                        "确认方案并生成全景图", variant="primary", elem_classes=["primary-action"]
                    )

            with gr.Tab("04 · 生成与验证", id=3):
                with gr.Row(equal_height=True):
                    validation_before = gr.Image(type="filepath", label="原始全景图", height=330)
                    validation_after = gr.Image(type="filepath", label="生成后的新全景图", height=330)
                backend_status = gr.Textbox(label="生成状态", interactive=False)
                with gr.Accordion("可选：选择修改后指标大表", open=False):
                    post_metrics_file = gr.Dropdown(
                        choices=table_choices,
                        value=None,
                        label="从后端选择修改后项目大表",
                        filterable=True,
                    )
                    with gr.Row():
                        post_metrics_sheet = gr.Dropdown(label="工作表", choices=[])
                        post_metrics_candidate = gr.Dropdown(label="候选指标行", choices=[])
                    post_metrics_status = gr.HTML(
                        status_html("尚未提供修改后指标，当前仅进行人工体验验证", "warn")
                    )
                    compare_table = gr.Dataframe(
                        headers=["指标", "原始值", "目标值", "修改后值", "目标偏差", "判定"],
                        interactive=False, elem_classes=["metric-grid"]
                    )
                _section_header("HUMAN VALIDATION", "修改后多人体验与专家评价")
                post_participants = gr.Dataframe(
                    value=[], headers=EXPERIENCE_HEADERS,
                    datatype=["str", "str", *(["number"] * 7)],
                    row_count=(0, "dynamic"), column_count=(9, "fixed"),
                    interactive=True, wrap=True,
                    label="可选；未填写也不会因缺少修改后指标而阻断保存",
                )
                with gr.Row():
                    expert_score = gr.Slider(1, 5, value=4, step=1, label="专家评分")
                    final_notes = gr.Textbox(label="专家备注", lines=3)
                with gr.Row():
                    rollback_input = gr.Button("返回重新设定体验目标")
                    rollback_morph = gr.Button("返回调整形态目标")
                    rollback_plan = gr.Button("返回调整空间布局方案")
                    finish_button = gr.Button("保存并结束", variant="primary")
                finish_message = gr.Textbox(label="保存结果", interactive=False)

        gr.HTML('<div class="footer-note">AI4City · 数据由项目大表驱动 · API Key 不进入页面状态</div>')

        refresh_catalog_button.click(
            refresh_backend_catalog,
            outputs=[
                image_selector,
                metrics_file,
                post_metrics_file,
                backend_catalog_status,
            ],
        )
        demo.load(
            load_sheet_choices,
            metrics_file,
            [metrics_sheet, metrics_status],
        ).then(
            preview_baseline,
            [image_selector, metrics_file, metrics_sheet, metrics_candidate],
            [baseline_table, metrics_candidate, metrics_status, image_name],
        )
        image_selector.change(
            select_backend_image,
            image_selector,
            [input_image, image_name],
        ).then(
            preview_baseline,
            [image_selector, metrics_file, metrics_sheet, metrics_candidate],
            [baseline_table, metrics_candidate, metrics_status, image_name],
        )
        metrics_file.change(
            load_sheet_choices, metrics_file, [metrics_sheet, metrics_status]
        ).then(
            preview_baseline,
            [image_selector, metrics_file, metrics_sheet, metrics_candidate],
            [baseline_table, metrics_candidate, metrics_status, image_name],
        )
        for component in (metrics_sheet, metrics_candidate):
            component.change(
                preview_baseline,
                [image_selector, metrics_file, metrics_sheet, metrics_candidate],
                [baseline_table, metrics_candidate, metrics_status, image_name],
            )

        confirm_input_button.click(
            confirm_inputs,
            [
                image_selector, metrics_file, metrics_sheet, metrics_candidate,
                space_type, current_use, main_people, time_weather, context_description,
                pre_participants, *experience_sliders,
            ],
            [
                browser_state, stepper, global_status, recovered_session_id,
                morph_image, morph_context, morph_people_count, morph_targets_json,
                morph_compare_table, translator_rationale, translator_evidence, *morph_sliders,
            ],
        )
        restore_agent_button.click(
            lambda state: reset_morph_sliders(state, "agent"), browser_state, morph_sliders
        )
        restore_baseline_button.click(
            lambda state: reset_morph_sliders(state, "baseline"), browser_state, morph_sliders
        )
        confirm_morph_button.click(
            confirm_morph,
            [browser_state, expert_note, *morph_sliders],
            [
                browser_state, stepper, global_status, plan_image, plan_metrics,
                plan_expert_note, action_table, plan_relations, plan_unchanged,
                plan_constraints, agent_prompt, final_prompt,
            ],
        )
        restore_prompt_button.click(restore_agent_prompt, browser_state, final_prompt)
        generate_button.click(
            generate_panorama,
            [browser_state, final_prompt, backend_selector],
            [
                browser_state, stepper, global_status, validation_before,
                validation_after, backend_status, post_metrics_status,
            ],
        )
        post_metrics_file.change(
            load_sheet_choices, post_metrics_file, [post_metrics_sheet, post_metrics_status]
        ).then(
            preview_post_metrics,
            [browser_state, post_metrics_file, post_metrics_sheet, post_metrics_candidate],
            [compare_table, post_metrics_candidate, post_metrics_status, browser_state],
        )
        for component in (post_metrics_sheet, post_metrics_candidate):
            component.change(
                preview_post_metrics,
                [browser_state, post_metrics_file, post_metrics_sheet, post_metrics_candidate],
                [compare_table, post_metrics_candidate, post_metrics_status, browser_state],
            )
        finish_button.click(
            finish_session,
            [browser_state, post_participants, expert_score, final_notes],
            [browser_state, stepper, global_status, finish_message],
        )
        rollback_input.click(
            lambda state: rollback_session(state, INPUT_PENDING), browser_state,
            [browser_state, stepper, global_status],
        )
        rollback_morph.click(
            lambda state: rollback_session(state, MORPH_REVIEW), browser_state,
            [browser_state, stepper, global_status],
        )
        rollback_plan.click(
            lambda state: rollback_session(state, PLAN_REVIEW), browser_state,
            [browser_state, stepper, global_status],
        )
        recovery_button.click(
            recover_session,
            recovery_id,
            [
                browser_state, stepper, global_status, image_selector, input_image, pre_participants,
                morph_compare_table, *morph_sliders, action_table, final_prompt,
                validation_before, validation_after, post_participants,
            ],
        )

    return demo


demo = build_ui()


def launch(**kwargs: Any):
    """以项目主题启动工作台；供脚本启动和外部集成复用。"""

    return demo.queue(default_concurrency_limit=4).launch(
        theme=APP_THEME,
        css=APP_CSS,
        **kwargs,
    )


if __name__ == "__main__":
    launch()
