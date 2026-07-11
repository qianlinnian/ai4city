from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import langchain_core
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change in a future version.*",
    category=LangChainPendingDeprecationWarning,
)
from langgraph.graph import END, START, StateGraph

from ai4city_mas.adapters import AdapterRegistry, default_adapters
from ai4city_mas.agents import (
    CAgent,
    DAgent,
    DatasetLoader,
    EAgent,
    ExpertGate,
    HAgent,
    XAgent,
    build_evidence_package,
)
from ai4city_mas.artifacts import ArtifactStore
from ai4city_mas.config import AppConfig
from ai4city_mas.domain import PipelineStage, PipelineState


@dataclass
class Runtime:
    config: AppConfig
    store: ArtifactStore
    adapters: AdapterRegistry


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        adapters: AdapterRegistry | None = None,
    ) -> None:
        self.config = config
        self.adapters = adapters or default_adapters()

    def run(self, run_id: str | None = None) -> PipelineState:
        resolved_run_id = run_id or _new_run_id()
        run_dir = self.config.paths.runs_dir / resolved_run_id
        store = ArtifactStore(run_dir, resolved_run_id)
        runtime = Runtime(self.config, store, self.adapters)
        graph = build_graph(runtime)
        initial: PipelineState = {
            "run_id": resolved_run_id,
            "run_dir": str(run_dir.resolve()),
            "stage": "not_started",
            "status": "running",
            "artifacts": {},
            "messages": [],
        }
        result = graph.invoke(initial)
        store.finalize(result["status"], result["messages"])
        return result


def build_graph(runtime: Runtime):
    config = runtime.config
    loader = DatasetLoader(config)
    e_agent = EAgent(runtime.adapters)
    expert = ExpertGate(config)
    c_agent = CAgent(config, runtime.adapters)
    d_agent = DAgent()
    x_agent = XAgent(config)
    h_agent = HAgent(config)

    def ingestion(state: PipelineState) -> dict[str, Any]:
        return _save(runtime, state, PipelineStage.DATA_INGESTION, loader.load())

    def perception(state: PipelineState) -> dict[str, Any]:
        scenes = runtime.store.read(PipelineStage.DATA_INGESTION)
        return _save(runtime, state, PipelineStage.PERCEPTION, e_agent.describe(scenes))

    def features(state: PipelineState) -> dict[str, Any]:
        descriptions = runtime.store.read(PipelineStage.PERCEPTION)
        return _save(
            runtime,
            state,
            PipelineStage.FEATURES,
            e_agent.extract_features(descriptions),
        )

    def expert_calibration(state: PipelineState) -> dict[str, Any]:
        payload = expert.calibrate(runtime.store.read(PipelineStage.FEATURES))
        status = "running" if payload["status"] == "approved" else "awaiting_expert"
        return _save(runtime, state, PipelineStage.EXPERT_CALIBRATION, payload, status)

    def context_matrix(state: PipelineState) -> dict[str, Any]:
        scenes = runtime.store.read(PipelineStage.DATA_INGESTION)
        return _save(
            runtime,
            state,
            PipelineStage.CONTEXT_MATRIX,
            c_agent.build_context_matrix(scenes),
        )

    def ai_scoring(state: PipelineState) -> dict[str, Any]:
        payload = c_agent.score(
            runtime.store.read(PipelineStage.FEATURES),
            runtime.store.read(PipelineStage.CONTEXT_MATRIX),
        )
        return _save(runtime, state, PipelineStage.AI_SCORING, payload)

    def interventions(state: PipelineState) -> dict[str, Any]:
        payload = d_agent.propose_interventions(
            runtime.store.read(PipelineStage.DATA_INGESTION),
            runtime.store.read(PipelineStage.FEATURES),
        )
        return _save(runtime, state, PipelineStage.INTERVENTIONS, payload)

    def vr_experiment(state: PipelineState) -> dict[str, Any]:
        payload = x_agent.build_experiment_plan(
            runtime.store.read(PipelineStage.INTERVENTIONS)
        )
        status = "running" if payload["ethics_approved"] else "awaiting_ethics"
        return _save(runtime, state, PipelineStage.VR_EXPERIMENT, payload, status)

    def human_validation(state: PipelineState) -> dict[str, Any]:
        payload = h_agent.validate(runtime.store.read(PipelineStage.AI_SCORING))
        can_continue = config.validation.allow_unvalidated_optimization
        status = "running" if can_continue else "awaiting_observed_data"
        return _save(runtime, state, PipelineStage.HUMAN_VALIDATION, payload, status)

    def parametric_optimization(state: PipelineState) -> dict[str, Any]:
        payload = d_agent.optimize(
            runtime.store.read(PipelineStage.INTERVENTIONS),
            runtime.store.read(PipelineStage.HUMAN_VALIDATION),
        )
        return _save(runtime, state, PipelineStage.PARAMETRIC_OPTIMIZATION, payload)

    def evidence_package(state: PipelineState) -> dict[str, Any]:
        payload = build_evidence_package(
            config.project.name,
            state["artifacts"],
            runtime.store.read(PipelineStage.HUMAN_VALIDATION),
            runtime.store.read(PipelineStage.PARAMETRIC_OPTIMIZATION),
        )
        return _save(
            runtime,
            state,
            PipelineStage.EVIDENCE_PACKAGE,
            payload,
            "completed",
        )

    graph = StateGraph(PipelineState)
    nodes: dict[PipelineStage, Callable[[PipelineState], dict[str, Any]]] = {
        PipelineStage.DATA_INGESTION: ingestion,
        PipelineStage.PERCEPTION: perception,
        PipelineStage.FEATURES: features,
        PipelineStage.EXPERT_CALIBRATION: expert_calibration,
        PipelineStage.CONTEXT_MATRIX: context_matrix,
        PipelineStage.AI_SCORING: ai_scoring,
        PipelineStage.INTERVENTIONS: interventions,
        PipelineStage.VR_EXPERIMENT: vr_experiment,
        PipelineStage.HUMAN_VALIDATION: human_validation,
        PipelineStage.PARAMETRIC_OPTIMIZATION: parametric_optimization,
        PipelineStage.EVIDENCE_PACKAGE: evidence_package,
    }
    for stage, function in nodes.items():
        graph.add_node(stage.value, function)

    graph.add_edge(START, PipelineStage.DATA_INGESTION.value)
    graph.add_edge(PipelineStage.DATA_INGESTION.value, PipelineStage.PERCEPTION.value)
    graph.add_edge(PipelineStage.PERCEPTION.value, PipelineStage.FEATURES.value)
    graph.add_edge(PipelineStage.FEATURES.value, PipelineStage.EXPERT_CALIBRATION.value)
    graph.add_conditional_edges(
        PipelineStage.EXPERT_CALIBRATION.value,
        _continue_or_stop,
        {"continue": PipelineStage.CONTEXT_MATRIX.value, "stop": END},
    )
    graph.add_edge(PipelineStage.CONTEXT_MATRIX.value, PipelineStage.AI_SCORING.value)
    graph.add_edge(PipelineStage.AI_SCORING.value, PipelineStage.INTERVENTIONS.value)
    graph.add_edge(PipelineStage.INTERVENTIONS.value, PipelineStage.VR_EXPERIMENT.value)
    graph.add_conditional_edges(
        PipelineStage.VR_EXPERIMENT.value,
        _continue_or_stop,
        {"continue": PipelineStage.HUMAN_VALIDATION.value, "stop": END},
    )
    graph.add_conditional_edges(
        PipelineStage.HUMAN_VALIDATION.value,
        _continue_or_stop,
        {"continue": PipelineStage.PARAMETRIC_OPTIMIZATION.value, "stop": END},
    )
    graph.add_edge(
        PipelineStage.PARAMETRIC_OPTIMIZATION.value,
        PipelineStage.EVIDENCE_PACKAGE.value,
    )
    graph.add_edge(PipelineStage.EVIDENCE_PACKAGE.value, END)
    return graph.compile()


def _save(
    runtime: Runtime,
    state: PipelineState,
    stage: PipelineStage,
    payload: Any,
    status: str = "running",
) -> dict[str, Any]:
    ref = runtime.store.write(stage, payload)
    stage_status = "completed" if status in {"running", "completed"} else status
    runtime.store.record(ref, stage_status=stage_status, pipeline_status=status)
    artifacts = dict(state["artifacts"])
    artifacts[stage.value] = ref.model_dump(mode="json")
    return {
        "stage": stage.value,
        "status": status,
        "artifacts": artifacts,
        "messages": [*state["messages"], f"{stage.value}: {status}"],
    }


def _continue_or_stop(state: PipelineState) -> str:
    return "continue" if state["status"] == "running" else "stop"


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"
