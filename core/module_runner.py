"""
Module runner — calls a sub-module's run.py via subprocess.
Passes inputs as JSON to stdin, reads UnifiedResult JSON from stdout.

Per-module Python executables are read from config/modules.yaml
(the optional `python_exe` field, relative to ROOT_DIR).
"""
from __future__ import annotations
import json
import subprocess
import sys
import os
from functools import lru_cache
from pathlib import Path

import yaml

from core.result_schema import UnifiedResult

ROOT_DIR = Path(__file__).resolve().parent.parent
MODULES_CONFIG = ROOT_DIR / "config" / "modules.yaml"

SCRIPT_MAP: dict[str, Path] = {
    "facettd":      ROOT_DIR / "module" / "facettd" / "run.py",
    "faceage":      ROOT_DIR / "module" / "FaceAge" / "run.py",
    "parkinsons":   ROOT_DIR / "module" / "Parkinsons" / "run.py",
    "lung_cancer":  ROOT_DIR / "module" / "Early_Stage_Lung_Cancer_Detection_from_Speech_Sounds" / "run.py",
    "mental_health": ROOT_DIR / "module" / "MentalHealth" / "run.py",
    "skin_disease":  ROOT_DIR / "module" / "SkinDisease" / "run.py",
    "facetrack":    ROOT_DIR / "module" / "FaceTrack" / "run.py",
    "voicetrack":   ROOT_DIR / "module" / "VoiceTrack" / "run.py",
}


@lru_cache(maxsize=1)
def _load_module_python_exes() -> dict[str, str]:
    """Return {module_id: absolute_python_exe} from modules.yaml."""
    exes: dict[str, str] = {}
    try:
        with open(MODULES_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for mod in cfg.get("modules", []):
            if "python_exe" in mod:
                exe = mod["python_exe"]
                # resolve relative paths against ROOT_DIR
                p = Path(exe) if Path(exe).is_absolute() else ROOT_DIR / exe
                exes[mod["id"]] = str(p)
    except Exception:
        pass
    return exes


def run_module(
    module_id: str,
    inputs: dict,
    python_exe: str | None = None,
) -> UnifiedResult:
    """
    Invoke module/{module_id}/run.py as a subprocess.

    Args:
        module_id:  One of "facettd", "faceage", "parkinsons", "lung_cancer", "mental_health", "skin_disease", "facetrack", "voicetrack".
        inputs:     Dict passed as JSON to the module via stdin.
        python_exe: Override Python interpreter. Falls back to modules.yaml
                    `python_exe`, then to the current interpreter.

    Returns:
        UnifiedResult instance.
    """
    script_path = SCRIPT_MAP.get(module_id)
    if script_path is None:
        return UnifiedResult(
            module_id=module_id,
            module_name=module_id,
            status="error",
            inputs=inputs,
            outputs={},
            summary=f"未知模块 ID：{module_id}",
            error=f"Unknown module id: {module_id}",
        )

    # Python exe priority: caller arg > modules.yaml > current interpreter
    exe = python_exe or _load_module_python_exes().get(module_id) or sys.executable

    try:
        proc = subprocess.run(
            [exe, str(script_path)],
            input=json.dumps(inputs, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
        )

        stdout = proc.stdout.strip()
        if not stdout:
            raise RuntimeError(
                f"模块无输出。\nstderr:\n{proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )

        # Find last JSON line (module may print debug info before the result)
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            raise RuntimeError(f"输出中未找到 JSON 对象。\nstdout:\n{stdout[-2000:]}")

        raw = json.loads(json_line)
        return UnifiedResult(**raw)

    except subprocess.TimeoutExpired:
        return UnifiedResult(
            module_id=module_id,
            module_name=module_id,
            status="error",
            inputs=inputs,
            outputs={},
            summary="模块运行超时（>300s）",
            error="TimeoutExpired",
        )
    except Exception as exc:
        return UnifiedResult(
            module_id=module_id,
            module_name=module_id,
            status="error",
            inputs=inputs,
            outputs={},
            summary=f"调用模块时出错：{exc}",
            error=str(exc),
        )
