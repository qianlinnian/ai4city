"""Optional FastAPI interface for the Excel-driven workflow.

The API never computes Task 1 metrics.  ``start_session`` receives the seven
baseline values selected from the project table, and post-edit values can be
submitted later through the quality endpoint.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.generation_backend import generate as generate_image
from config import ASSETS_DIR, SESSION_DIR, UPLOAD_DIR
from pipeline.orchestrator import PipelineOrchestrator
from utils.scene_data import list_scene_choices, load_scene_bundle


app = FastAPI(title="Panorama Multi-Agent API", version="0.4.0")
pipe = PipelineOrchestrator()


class ExperienceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comfort: float = Field(..., ge=1, le=5)
    naturalness: float = Field(..., ge=1, le=5)
    safety: float = Field(..., ge=1, le=5)
    relaxation: float = Field(..., ge=1, le=5)
    environmental_disturbance: float = Field(..., ge=1, le=5)
    stay_intention: float = Field(..., ge=1, le=5)
    overall_impression: float = Field(..., ge=1, le=5)


class RunTranslatorIn(BaseModel):
    session_id: str
    experience_targets: ExperienceIn
    experience_records: Optional[list[dict[str, Any]]] = None
    experience_baseline: Optional[dict[str, float]] = None


class ConfirmMorphIn(BaseModel):
    session_id: str
    human_metrics: Optional[dict[str, float]] = None
    note: str = ""
    language: str = "en"


class ConfirmPlanIn(BaseModel):
    session_id: str
    human_plan: str


class GenerateIn(BaseModel):
    session_id: str
    backend: Optional[str] = None


class QualityMetricsIn(BaseModel):
    session_id: str
    modified_metrics: dict[str, float]
    thresholds: Optional[dict[str, float]] = None


class PostExperienceIn(BaseModel):
    session_id: str
    post_edit_experience: list[dict[str, Any]]


class MemoryIn(BaseModel):
    session_id: str
    human_corrected_metrics: Optional[dict[str, float]] = None
    score: Optional[float] = None
    notes: str = ""


def _load_state(session_id: str) -> dict[str, Any]:
    if not session_id or not session_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "session_id 格式无效")
    path = SESSION_DIR / f"{session_id}.json"
    if not path.is_file():
        raise HTTPException(404, "找不到会话")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "会话数据无法读取") from exc
    if not isinstance(state, dict) or state.get("session_id") != session_id:
        raise HTTPException(500, "会话数据格式错误")
    return state


def _bad_request(operation):
    try:
        return operation()
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/health")
def health():
    return {"ok": True, "version": "0.4.0", "task1_mode": "offline_table"}


@app.post("/extract_metrics", status_code=410)
def extract_metrics_removed():
    raise HTTPException(
        410,
        "在线指标提取已移除；请离线运行 Task 1，并把大表格中的七项指标传给 start_session。",
    )


@app.get("/scenes")
def list_scenes():
    """列出可选场景（assets 文件名，或边缘图 stem）。"""
    return {"assets_dir": str(ASSETS_DIR), "scenes": list_scene_choices()}


@app.get("/scenes/{image_name}")
def get_scene(image_name: str):
    """按图片名加载 Excel 指标 + 分析图路径。"""
    try:
        return load_scene_bundle(image_name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e


class StartFromDatasetIn(BaseModel):
    image_name: str
    scene_context: Optional[dict[str, Any]] = None
    pre_edit_experience: Optional[list[dict[str, Any]]] = None


@app.post("/pipeline/start_from_dataset")
def pipeline_start_from_dataset(body: StartFromDatasetIn):
    """从 assets + filled_metrics.xlsx 启动 session（不重跑形态解析）。"""
    try:
        bundle = load_scene_bundle(body.image_name)
    except Exception as e:
        raise HTTPException(400, str(e)) from e

    images = bundle.get("images") or {}
    original = images.get("original") or images.get("edge_map") or images.get("seg_map")
    if not original:
        raise HTTPException(400, f"找不到原图或分析图: {body.image_name}")

    scene = (
        body.scene_context
        if body.scene_context is not None
        else bundle["scene_context"]
    )
    pre_edit = (
        body.pre_edit_experience
        if body.pre_edit_experience is not None
        else bundle["persons"]
    )
    state = pipe.start_session(
        original,
        bundle["morph_metrics"],
        scene_context=scene,
        pre_edit_experience=pre_edit,
    )
    return state


@app.post("/pipeline/start_session")
async def pipeline_start_session(
    file: UploadFile = File(...),
    baseline_metrics_json: str = Form(...),
    scene_context_json: str = Form("{}"),
    pre_edit_experience_json: str = Form("[]"),
):
    try:
        baseline = json.loads(baseline_metrics_json)
        scene = json.loads(scene_context_json or "{}")
        pre_edit = json.loads(pre_edit_experience_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"JSON 格式错误：{exc.msg}") from exc
    if not isinstance(baseline, dict):
        raise HTTPException(400, "baseline_metrics_json 必须是 JSON 对象")
    if not isinstance(scene, dict) or not isinstance(pre_edit, list):
        raise HTTPException(400, "情景要素必须是对象，体验记录必须是数组")

    suffix = Path(file.filename or "pano.jpg").suffix or ".jpg"
    path = UPLOAD_DIR / f"api_{uuid.uuid4().hex}{suffix}"
    try:
        with path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        return _bad_request(
            lambda: pipe.start_session(
                path,
                baseline,
                scene_context=scene,
                pre_edit_experience=pre_edit,
            )
        )
    except HTTPException:
        path.unlink(missing_ok=True)
        raise


@app.get("/pipeline/session/{session_id}")
def pipeline_get_session(session_id: str):
    return _load_state(session_id)


@app.post("/pipeline/run_translator")
def pipeline_run_translator(body: RunTranslatorIn):
    state = _load_state(body.session_id)
    return _bad_request(
        lambda: pipe.run_translator(
            state,
            body.experience_targets.model_dump(),
            experience_records=body.experience_records,
            experience_baseline=body.experience_baseline,
        )
    )


@app.post("/pipeline/confirm_morph")
def pipeline_confirm_morph(body: ConfirmMorphIn):
    state = _load_state(body.session_id)
    return _bad_request(
        lambda: pipe.confirm_morph(
            state,
            human_metrics=body.human_metrics,
            note=body.note,
            language=body.language,
        )
    )


@app.post("/pipeline/confirm_plan")
def pipeline_confirm_plan(body: ConfirmPlanIn):
    state = _load_state(body.session_id)
    return _bad_request(lambda: pipe.confirm_plan(state, body.human_plan))


@app.post("/pipeline/generate")
def pipeline_generate(body: GenerateIn):
    state = _load_state(body.session_id)

    def operation():
        if state.get("stage") != "plan_confirmed":
            raise ValueError("当前阶段不允许生成；请先确认空间布局方案")
        result = generate_image(
            state["image_path"],
            state.get("final_prompt") or "",
            backend=body.backend,
        )
        return pipe.record_generation(state, result)

    return _bad_request(operation)


@app.post("/pipeline/quality_metrics")
def pipeline_quality_metrics(body: QualityMetricsIn):
    state = _load_state(body.session_id)
    return _bad_request(
        lambda: pipe.record_quality_metrics(
            state,
            body.modified_metrics,
            thresholds=body.thresholds,
        )
    )


@app.post("/pipeline/post_experience")
def pipeline_post_experience(body: PostExperienceIn):
    state = _load_state(body.session_id)
    return _bad_request(
        lambda: pipe.record_post_experience(state, body.post_edit_experience)
    )


@app.post("/pipeline/memory")
def pipeline_memory(body: MemoryIn):
    state = _load_state(body.session_id)
    return _bad_request(
        lambda: pipe.save_memory(
            state,
            human_corrected_metrics=body.human_corrected_metrics,
            score=body.score,
            notes=body.notes,
        )
    )


@app.post("/pipeline/start", status_code=410)
def pipeline_start_legacy_removed():
    raise HTTPException(
        410,
        "旧版入口依赖在线 Task 1，现已移除；请使用 /pipeline/start_session。",
    )
