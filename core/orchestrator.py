"""
Orchestrator — runs enabled modules in parallel, then renders the prompt.
"""
from __future__ import annotations
import concurrent.futures
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader, Undefined
import yaml

from core.module_runner import run_module
from core.result_schema import UnifiedResult

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT_DIR / "config" / "prompt_template.yaml"

IMAGE_MODULES = {"facettd", "faceage", "mental_health", "skin_disease"}
AUDIO_MODULES = {"parkinsons", "lung_cancer"}
VIDEO_MODULES: set = set()  # FaceTrack uses video_path (handled by dedicated API)


class _SilentUndefined(Undefined):
    """Jinja2 Undefined that renders as empty string instead of raising."""

    def __str__(self):
        return ""

    def __getattr__(self, _):
        return self


def _build_inputs(
    module_id: str, image_path: str | None, audio_path: str | None, age: int
) -> dict[str, Any]:
    inputs: dict[str, Any] = {"age": age}
    if module_id in IMAGE_MODULES and image_path:
        inputs["image_path"] = image_path
    if module_id in AUDIO_MODULES and audio_path:
        inputs["audio_path"] = audio_path
    return inputs


def load_prompt_template() -> str:
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("template", "")


def save_prompt_template(template_str: str) -> None:
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["template"] = template_str
    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, width=120)


def render_prompt(
    results: dict[str, UnifiedResult],
    age: int,
    template_str: str | None = None,
) -> str:
    if template_str is None:
        template_str = load_prompt_template()

    env = Environment(loader=BaseLoader(), undefined=_SilentUndefined)
    tmpl = env.from_string(template_str)

    ctx: dict[str, Any] = {
        "age": age,
        "results": {mid: r.to_template_dict() for mid, r in results.items()},
    }
    for mid, r in results.items():
        ctx[mid] = r.to_template_dict()

    return tmpl.render(**ctx)


def orchestrate(
    enabled_modules: list[str],
    image_path: str | None,
    audio_path: str | None,
    age: int,
    python_exe: str | None = None,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    """
    Run all enabled modules concurrently, then render the prompt.

    Returns:
        {
            "results": { module_id: UnifiedResult.dict() },
            "prompt": "rendered prompt string",
        }
    """
    results: dict[str, UnifiedResult] = {}

    def _run(module_id: str) -> tuple[str, UnifiedResult]:
        inputs = _build_inputs(module_id, image_path, audio_path, age)
        return module_id, run_module(module_id, inputs, python_exe)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run, mid): mid for mid in enabled_modules}
        for future in concurrent.futures.as_completed(futures):
            mid, result = future.result()
            results[mid] = result

    prompt = render_prompt(results, age, template_str=prompt_override)

    return {
        "results": {mid: r.model_dump() for mid, r in results.items()},
        "prompt": prompt,
    }
