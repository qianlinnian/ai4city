"""
可选 FastAPI 接口（组员也可不用 Gradio，直接 HTTP 联调）v2

启动:
  cd code
  uvicorn app.api_server:app --reload --port 8000

主要端点:
  POST /extract_metrics              仅形态解析（SegFormer / fallback）
  POST /pipeline/start_session       上传+情景+解析
  POST /pipeline/run_translator      体验滑块 → 翻译官
  POST /pipeline/confirm_morph       人工确认形态 → 制图员
  POST /pipeline/confirm_plan        人工润色自然语言方案
  POST /pipeline/generate            文生图+质检
  POST /pipeline/post_experience     修改后多人体验
  POST /pipeline/memory              入库
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

from config import ASSETS_DIR, UPLOAD_DIR
from morph_metrics_extractor import MorphMetricsExtractor
from pipeline.orchestrator import PipelineOrchestrator
from utils.scene_data import list_scene_choices, load_scene_bundle

app = FastAPI(title="Micro-Space Multi-Agent API", version="0.3.0")
pipe = PipelineOrchestrator(force_metrics_fallback=True)
extractor = MorphMetricsExtractor(force_fallback=True)
SESSIONS: dict[str, dict] = {}


class ExperienceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comfort: float = Field(..., ge=1, le=5)
    naturalness: float = Field(..., ge=1, le=5)
    safety: float = Field(..., ge=1, le=5)
    relaxation: float = Field(..., ge=1, le=5)
    environmental_disturbance: float = Field(..., ge=1, le=5)
    stay_intention: float = Field(..., ge=1, le=5)
    overall_impression: float = Field(..., ge=1, le=5)


class SceneContextIn(BaseModel):
    observation_time: str = ""
    observation_weather: str = ""
    people_flow: str = ""
    space_type: str = ""
    sound_type: str = ""
    maintenance_status: str = ""
    traffic_flow: str = ""
    description: str = ""


class StartSessionIn(BaseModel):
    scene_context: SceneContextIn = Field(default_factory=SceneContextIn)
    pre_edit_experience: list[dict[str, Any]] = Field(default_factory=list)


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


class PostExperienceIn(BaseModel):
    session_id: str
    post_edit_experience: list[dict[str, Any]]


class MemoryIn(BaseModel):
    session_id: str
    human_corrected_metrics: Optional[dict[str, float]] = None
    score: Optional[float] = None
    notes: str = ""


@app.get("/health")
def health():
    return {"ok": True, "version": "0.3.0"}


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
    scene_context: Optional[SceneContextIn] = None
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
        body.scene_context.model_dump()
        if body.scene_context is not None
        else bundle["scene_context"]
    )
    pre_edit = (
        body.pre_edit_experience
        if body.pre_edit_experience is not None
        else bundle["persons"]
    )
    image_name = Path(images["original"]).name if images.get("original") else body.image_name
    state = pipe.start_session(
        original,
        scene_context=scene,
        pre_edit_experience=pre_edit,
        baseline_metrics=bundle["morph_metrics"],
        image_name=image_name,
        skip_extract=True,
    )
    state["dataset_images"] = images
    SESSIONS[state["session_id"]] = state
    return state


@app.post("/extract_metrics")
async def extract_metrics(file: UploadFile = File(...), fallback: bool = True):
    suffix = Path(file.filename or "pano.jpg").suffix or ".jpg"
    path = UPLOAD_DIR / f"api_{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        f.write(await file.read())
    ex = MorphMetricsExtractor(force_fallback=fallback)
    metrics = ex.calculate(path)
    return {"image_path": str(path), "metrics": metrics.as_dict(), "display": metrics.as_percent_display()}


@app.post("/pipeline/start_session")
async def pipeline_start_session(
    file: UploadFile = File(...),
    scene_context_json: str = Form("{}"),
    pre_edit_experience_json: str = Form("[]"),
):
    suffix = Path(file.filename or "pano.jpg").suffix or ".jpg"
    path = UPLOAD_DIR / f"api_{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        scene = json.loads(scene_context_json or "{}")
        pre_edit = json.loads(pre_edit_experience_json or "[]")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"JSON invalid: {e}") from e

    state = pipe.start_session(path, scene_context=scene, pre_edit_experience=pre_edit)
    SESSIONS[state["session_id"]] = state
    return state


@app.post("/pipeline/run_translator")
def pipeline_run_translator(body: RunTranslatorIn):
    state = SESSIONS.get(body.session_id)
    if not state:
        raise HTTPException(404, "session not found")
    state = pipe.run_translator(
        state,
        body.experience_targets.model_dump(),
        experience_records=body.experience_records,
        experience_baseline=body.experience_baseline,
    )
    SESSIONS[body.session_id] = state
    return state


@app.post("/pipeline/confirm_morph")
def pipeline_confirm_morph(body: ConfirmMorphIn):
    state = SESSIONS.get(body.session_id)
    if not state:
        raise HTTPException(404, "session not found")
    state = pipe.confirm_morph(
        state,
        human_metrics=body.human_metrics,
        note=body.note,
        language=body.language,
    )
    SESSIONS[body.session_id] = state
    return state


@app.post("/pipeline/confirm_plan")
def pipeline_confirm_plan(body: ConfirmPlanIn):
    state = SESSIONS.get(body.session_id)
    if not state:
        raise HTTPException(404, "session not found")
    state = pipe.confirm_plan(state, body.human_plan)
    SESSIONS[body.session_id] = state
    return state


@app.post("/pipeline/generate")
def pipeline_generate(body: GenerateIn):
    state = SESSIONS.get(body.session_id)
    if not state:
        raise HTTPException(404, "session not found")
    state = pipe.generate_and_check(state)
    SESSIONS[body.session_id] = state
    return state


@app.post("/pipeline/post_experience")
def pipeline_post_experience(body: PostExperienceIn):
    state = SESSIONS.get(body.session_id)
    if not state:
        raise HTTPException(404, "session not found")
    state = pipe.record_post_experience(state, body.post_edit_experience)
    SESSIONS[body.session_id] = state
    return state


@app.post("/pipeline/memory")
def pipeline_memory(body: MemoryIn):
    state = SESSIONS.get(body.session_id)
    if not state:
        raise HTTPException(404, "session not found")
    state = pipe.save_memory(
        state,
        human_corrected_metrics=body.human_corrected_metrics,
        score=body.score,
        notes=body.notes,
    )
    SESSIONS[body.session_id] = state
    return state


# ---------- 兼容旧端点（deprecated）----------
@app.post("/pipeline/start")
async def pipeline_start_legacy(
    file: UploadFile = File(...),
    comfort: float = Form(4),
    restoration: float = Form(5),
    safety: float = Form(3),
    pleasure: float = Form(4),
    stay: float = Form(4),
):
    """@deprecated 请使用 start_session + run_translator"""
    suffix = Path(file.filename or "pano.jpg").suffix or ".jpg"
    path = UPLOAD_DIR / f"api_{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    state = pipe.start_session(path)
    state = pipe.run_translator(
        state,
        {
            "comfort": comfort,
            "naturalness": 3,
            "restoration": restoration,
            "safety": safety,
            "environmental_disturbance": 3,
            "pleasure": pleasure,
            "stay": stay,
        },
        experience_baseline={
            "comfort": 3,
            "naturalness": 3,
            "safety": 3,
            "relaxation": 3,
            "environmental_disturbance": 3,
            "stay_intention": 3,
            "overall_impression": 3,
        },
    )
    SESSIONS[state["session_id"]] = state
    return state
