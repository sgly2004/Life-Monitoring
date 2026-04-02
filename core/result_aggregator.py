"""
Result aggregator — runs health modules against a PreprocessedSample
and produces a SampleAnalysisResult.

Module calling conventions:
  FaceAge:    {"image_paths": [...]}  →  list of UnifiedResult (batch mode)
  facettd:    {"image_path": "...", "age": N}  →  one UnifiedResult per image
  Parkinsons: {"audio_path": "..."}   →  one UnifiedResult per audio clip
  lung_cancer:{"audio_path": "..."}   →  one UnifiedResult per audio clip
  SkinDisease:{"image_path": "..."}   →  one UnifiedResult per image (optional)

Aggregation:
  - FaceAge:    median across all successful predictions
  - facettd:    median remaining-life across all successful predictions
  - Parkinsons: mean probability + majority-vote label across all clips
  - lung_cancer:same
  - SkinDisease:most-common label across all images

Progress:
  progress_cb(event_type, **kwargs) is called after each item finishes.
  Useful for streaming per-model progress to the frontend.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.module_runner import _load_module_python_exes, SCRIPT_MAP
from core.workflow_types import (
    DiseaseRiskResult,
    FaceAgeResult,
    FacettdResult,
    PreprocessedSample,
    SampleAnalysisResult,
)

logger = logging.getLogger(__name__)


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _call_module(module_id: str, inputs: dict, timeout: int = 300) -> dict:
    """
    Call a module's run.py subprocess with JSON stdin.
    Returns the parsed last-JSON-line from stdout.
    Raises RuntimeError on failure.
    """
    script = SCRIPT_MAP.get(module_id)
    if script is None:
        raise ValueError(f"Unknown module id: {module_id}")

    exe = _load_module_python_exes().get(module_id) or sys.executable

    proc = subprocess.run(
        [exe, str(script)],
        input=json.dumps(inputs, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
    )

    stdout = proc.stdout.strip()
    json_line = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            json_line = line
            break

    if json_line is None:
        raise RuntimeError(
            f"[{module_id}] 无 JSON 输出。\nstderr:\n{proc.stderr[-1500:] if proc.stderr else '(empty)'}"
        )

    return json.loads(json_line)


# ── FaceAge ───────────────────────────────────────────────────────────────────

def _run_faceage(
    image_paths: list[str],
    progress_cb: Optional[Callable] = None,
) -> FaceAgeResult:
    """
    Run FaceAge in batch mode across all face images.
    Returns aggregated FaceAgeResult (median biological age).
    """
    if not image_paths:
        return FaceAgeResult(status="skipped", error="No face images available")

    try:
        raw = _call_module("faceage", {"image_paths": image_paths}, timeout=600)
    except Exception as exc:
        if progress_cb:
            progress_cb("model_progress", model="faceage", done=True, status="error",
                        desc=f"⚠ FaceAge 失败：{exc}")
        return FaceAgeResult(status="error", error=str(exc))

    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    bio_ages: list[float] = []
    for item in items:
        if item.get("status") == "success":
            bio_age = item.get("outputs", {}).get("biological_age")
            if bio_age is not None:
                bio_ages.append(float(bio_age))

    if not bio_ages:
        if progress_cb:
            progress_cb("model_progress", model="faceage", done=True, status="error",
                        desc="⚠ FaceAge：所有图像均未能预测生物年龄")
        return FaceAgeResult(status="error", error="所有图像均未能成功预测生物年龄")

    median_age = statistics.median(bio_ages)
    if progress_cb:
        progress_cb("model_progress", model="faceage", done=True, status="ok",
                    value=round(median_age, 1),
                    desc=f"✓ FaceAge 完成，生物年龄中位数 {median_age:.1f} 岁（共 {len(bio_ages)} 张）")
    return FaceAgeResult(
        biological_age_median=median_age,
        biological_age_values=bio_ages,
        status="ok",
    )


# ── facettd ───────────────────────────────────────────────────────────────────

def _run_facettd_one(image_path: str, age: int) -> Optional[float]:
    """Run facettd for one image. Returns remaining life in years or None."""
    try:
        raw = _call_module("facettd", {"image_path": image_path, "age": age})
        if raw.get("status") == "success":
            ttd = raw.get("outputs", {}).get("time_to_death_years")
            if ttd is not None:
                return float(ttd)
    except Exception as exc:
        logger.debug("facettd failed for %s: %s", image_path, exc)
    return None


def _run_facettd(
    image_paths: list[str],
    age: int,
    progress_cb: Optional[Callable] = None,
) -> FacettdResult:
    """
    Run facettd on each face image, aggregate with median.
    Emits per-image progress via progress_cb.
    """
    if not image_paths:
        return FacettdResult(status="skipped", error="No face images available")

    total = len(image_paths)
    completed_box = [0]
    lock = threading.Lock()

    def run_one_tracked(path: str) -> Optional[float]:
        result = _run_facettd_one(path, age)
        with lock:
            completed_box[0] += 1
            c = completed_box[0]
        if progress_cb:
            progress_cb("model_progress", model="facettd",
                        current=c, total=total, done=(c == total),
                        desc=f"facettd: {c}/{total} 张")
        return result

    values: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_one_tracked, image_paths))
        values = [r for r in results if r is not None]

    if not values:
        return FacettdResult(status="error", error="facettd 未返回有效结果")

    median_val = statistics.median(values)
    return FacettdResult(
        remaining_life_median=median_val,
        remaining_life_values=values,
        status="ok",
    )


# ── Disease modules (Parkinsons / lung_cancer / skin_disease) ─────────────────

def _run_audio_disease_module(
    module_id: str,
    audio_paths: list[str],
    prob_key: str = "probability",
    label_key: str = "label",
    progress_cb: Optional[Callable] = None,
) -> DiseaseRiskResult:
    """
    Generic aggregator for audio-based disease detection.
    Runs module once per audio clip, aggregates mean probability + majority vote.
    Emits per-clip progress via progress_cb.
    """
    if not audio_paths:
        return DiseaseRiskResult(
            module_id=module_id,
            status="skipped",
            error="No audio segments available",
        )

    total = len(audio_paths)
    probs: list[float] = []
    labels: list[str] = []

    for i, audio_path in enumerate(audio_paths):
        try:
            raw = _call_module(module_id, {"audio_path": audio_path})
            if raw.get("status") == "success":
                p = raw.get("outputs", {}).get(prob_key)
                l = raw.get("outputs", {}).get(label_key)
                if p is not None:
                    probs.append(float(p))
                if l is not None:
                    labels.append(str(l))
        except Exception as exc:
            logger.debug("[%s] failed for %s: %s", module_id, audio_path, exc)
        if progress_cb:
            progress_cb("model_progress", model=module_id,
                        current=i + 1, total=total, done=(i + 1 == total),
                        desc=f"{module_id}: {i + 1}/{total} 段")

    if not probs and not labels:
        return DiseaseRiskResult(module_id=module_id, status="error", error="无有效结果")

    mean_prob = statistics.mean(probs) if probs else None
    vote_counts: dict[str, int] = {}
    for lb in labels:
        vote_counts[lb] = vote_counts.get(lb, 0) + 1
    majority_label = max(vote_counts, key=vote_counts.__getitem__) if vote_counts else None

    return DiseaseRiskResult(
        module_id=module_id,
        risk_label=majority_label,
        risk_probability=mean_prob,
        vote_counts=vote_counts,
        status="ok",
    )


def _run_image_disease_module(
    module_id: str,
    image_paths: list[str],
    prob_key: str = "probability",
    label_key: str = "label",
    progress_cb: Optional[Callable] = None,
) -> DiseaseRiskResult:
    """
    Generic aggregator for image-based disease detection (e.g. SkinDisease).
    Emits per-image progress via progress_cb.
    """
    if not image_paths:
        return DiseaseRiskResult(
            module_id=module_id,
            status="skipped",
            error="No face images available",
        )

    total = len(image_paths)
    probs: list[float] = []
    labels: list[str] = []

    for i, image_path in enumerate(image_paths):
        try:
            raw = _call_module(module_id, {"image_path": image_path})
            if raw.get("status") == "success":
                p = raw.get("outputs", {}).get(prob_key)
                l = raw.get("outputs", {}).get(label_key)
                if p is not None:
                    probs.append(float(p))
                if l is not None:
                    labels.append(str(l))
        except Exception as exc:
            logger.debug("[%s] failed for %s: %s", module_id, image_path, exc)
        if progress_cb:
            progress_cb("model_progress", model=module_id,
                        current=i + 1, total=total, done=(i + 1 == total),
                        desc=f"{module_id}: {i + 1}/{total} 张")

    if not probs and not labels:
        return DiseaseRiskResult(module_id=module_id, status="error", error="无有效结果")

    mean_prob = statistics.mean(probs) if probs else None
    vote_counts: dict[str, int] = {}
    for lb in labels:
        vote_counts[lb] = vote_counts.get(lb, 0) + 1
    majority_label = max(vote_counts, key=vote_counts.__getitem__) if vote_counts else None

    return DiseaseRiskResult(
        module_id=module_id,
        risk_label=majority_label,
        risk_probability=mean_prob,
        vote_counts=vote_counts,
        status="ok",
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def aggregate_sample(
    sample: PreprocessedSample,
    include_skin_disease: bool = False,
    progress_cb: Optional[Callable] = None,
) -> SampleAnalysisResult:
    """
    Run all health modules against a preprocessed sample and aggregate results.

    Modules run:
        - FaceAge       (batch, all face images)
        - facettd       (per image, parallel, using age_at_capture)
        - Parkinsons    (per audio segment)
        - lung_cancer   (per audio segment)
        - SkinDisease   (optional, per face image)

    Args:
        sample:               Output of preprocess_pipeline.preprocess_sample.
        include_skin_disease: Whether to include the SkinDisease module.
        progress_cb:          Optional callable(event_type, **data) for progress.
    """
    face_paths = sample.face_image_paths
    audio_paths = sample.audio_segment_paths
    age = sample.age_at_capture

    logger.info("[%s] 开始聚合分析（%d 张脸，%d 段音频，年龄 %d）",
                sample.subject_name, len(face_paths), len(audio_paths), age)

    if progress_cb:
        progress_cb("phase", phase="models",
                    faces=len(face_paths), audio_segs=len(audio_paths),
                    desc=f"正在运行健康分析模型（{len(face_paths)} 张图像，{len(audio_paths)} 段音频）")

    # Run all modules in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        f_faceage = executor.submit(_run_faceage, face_paths, progress_cb)
        f_facettd = executor.submit(_run_facettd, face_paths, age, progress_cb)
        f_parkinson = executor.submit(
            _run_audio_disease_module, "parkinsons", audio_paths,
            "prob_parkinson", "label", progress_cb,
        )
        f_lung = executor.submit(
            _run_audio_disease_module, "lung_cancer", audio_paths,
            "probability", "label", progress_cb,
        )
        f_skin = (
            executor.submit(
                _run_image_disease_module, "skin_disease", face_paths,
                "probability", "label", progress_cb,
            )
            if include_skin_disease else None
        )

        faceage_result = f_faceage.result()
        facettd_result = f_facettd.result()
        parkinson_result = f_parkinson.result()
        lung_result = f_lung.result()
        skin_result = f_skin.result() if f_skin is not None else None

    if (
        faceage_result.biological_age_median is not None
        and faceage_result.status == "ok"
    ):
        faceage_result.age_delta = faceage_result.biological_age_median - age

    logger.info(
        "[%s] 聚合完成 — FaceAge: %s, facettd: %s, Parkinsons: %s, LungCancer: %s",
        sample.subject_name,
        faceage_result.status,
        facettd_result.status,
        parkinson_result.status,
        lung_result.status,
    )

    return SampleAnalysisResult(
        subject_name=sample.subject_name,
        captured_at=sample.captured_at,
        age_at_capture=age,
        face_age=faceage_result,
        facettd=facettd_result,
        parkinsons=parkinson_result,
        lung_cancer=lung_result,
        skin_disease=skin_result,
    )
