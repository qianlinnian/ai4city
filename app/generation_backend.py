"""Unified, lazy image-generation backend used by the UI.

The public :func:`generate` entry point deliberately accepts only an input
image, a prompt, and an optional backend name.  Credentials remain in the
environment/configuration layer and are never returned to callers.
"""

from __future__ import annotations

import importlib
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from schemas.models import GenerationResult


BackendName = Literal["mock", "seedream", "worldlabs"]

_BACKEND_ALIASES: dict[str, BackendName] = {
    "mock": "mock",
    "seedream": "seedream",
    "seed-dream": "seedream",
    "worldlabs": "worldlabs",
    "world-labs": "worldlabs",
    "world_labs": "worldlabs",
}


def _normalise_backend(value: str) -> BackendName:
    key = value.strip().lower()
    try:
        return _BACKEND_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted({*(_BACKEND_ALIASES.values())}))
        raise ValueError(f"未知生成后端 {value!r}；可选值：{supported}") from exc


def default_backend() -> BackendName:
    """Resolve the configured backend without exposing any credential.

    ``GENERATION_BACKEND`` (or the legacy ``IMAGE_GENERATION_BACKEND``) wins.
    With no explicit choice, mock mode is honoured first; otherwise a
    configured Seedream key is preferred, followed by World Labs.  A project
    without credentials always remains safely in local mock mode.
    """

    explicit = os.getenv("GENERATION_BACKEND", "").strip()
    if not explicit:
        explicit = os.getenv("IMAGE_GENERATION_BACKEND", "").strip()
    if explicit:
        return _normalise_backend(explicit)

    config = importlib.import_module("config")
    configured = str(getattr(config, "GENERATION_BACKEND", "")).strip()
    if configured:
        return _normalise_backend(configured)

    if str(getattr(config, "RUN_MODE", "auto")).lower() == "mock":
        return "mock"
    if bool(getattr(config, "SEEDREAM_API_KEY", "")):
        return "seedream"
    if bool(getattr(config, "WORLDLABS_API_KEY", "")):
        return "worldlabs"
    return "mock"


def _validate_input(image_path: str | Path, prompt: str) -> tuple[Path, str]:
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入图片：{source}")
    clean_prompt = str(prompt).strip()
    if not clean_prompt:
        raise ValueError("生成提示词不能为空")
    return source, clean_prompt


def _mock_generate(image_path: Path, prompt: str) -> "GenerationResult":
    # Imports stay inside the selected path so importing the UI never loads
    # model/API integrations unnecessarily.
    config = importlib.import_module("config")
    models = importlib.import_module("schemas.models")
    target_dir = Path(getattr(config, "TARGET_IMG_DIR"))
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = image_path.suffix or ".png"
    output = target_dir / f"{image_path.stem}_mock_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.copy2(image_path, output)
    return models.GenerationResult(
        output_image_path=str(output),
        prompt_used=prompt,
        mock=True,
        raw={"backend": "mock", "status": "mock"},
    )


def _agent_generate(
    image_path: Path,
    prompt: str,
    backend: Literal["seedream", "worldlabs"],
) -> "GenerationResult":
    if backend == "seedream":
        module_name, class_name = "agents.seedream_agent", "SeedreamAgent"
    else:
        module_name, class_name = "agents.worldlabs_agent", "WorldLabsAgent"

    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    result = agent_class().run(image_path=image_path, prompt=prompt)
    result.raw.setdefault("backend", backend)
    result.raw.setdefault("status", "mock" if result.mock else "live")
    return result


def generate(
    image_path: str | Path,
    prompt: str,
    backend: str | None = None,
) -> "GenerationResult":
    """Generate an edited image through Mock, Seedream, or World Labs.

    The two live integrations retain their existing fallback behaviour.  If
    they fall back locally, the returned ``GenerationResult.mock`` flag and
    ``raw['status']`` make that state visible without revealing an API key.
    """

    source, clean_prompt = _validate_input(image_path, prompt)
    selected = default_backend() if backend is None else _normalise_backend(backend)
    if selected == "mock":
        return _mock_generate(source, clean_prompt)
    return _agent_generate(source, clean_prompt, selected)


__all__ = ["BackendName", "default_backend", "generate"]
