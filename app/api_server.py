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
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import UPLOAD_DIR
from morph_metrics_extractor import MorphMetricsExtractor
from pipeline.orchestrator import PipelineOrchestrator

app = FastAPI(title="Micro-Space Multi-Agent API", version="0.2.0")
pipe = PipelineOrchestrator(force_metrics_fallback=True)
extractor = MorphMetricsExtractor(force_fallback=True)
SESSIONS: dict[str, dict] = {}


class ExperienceIn(BaseModel):
    comfort: float = 4
    restoration: float = 5
    safety: float = 3
    pleasure: float = 4
    stay: float = 4


class SceneContextIn(BaseModel):
    location_type: str = ""
    time_of_day: str = ""
    weather: str = ""
    crowd_level: str = ""
    description: str = ""


class StartSessionIn(BaseModel):
    scene_context: SceneContextIn = Field(default_factory=SceneContextIn)
    pre_edit_experience: list[dict[str, Any]] = Field(default_factory=list)


class RunTranslatorIn(BaseModel):
    session_id: str
    experience_targets: ExperienceIn
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
    return {"ok": True, "version": "0.2.0"}


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
        {"comfort": comfort, "restoration": restoration, "safety": safety,
         "pleasure": pleasure, "stay": stay},
    )
    SESSIONS[state["session_id"]] = state
    return state
