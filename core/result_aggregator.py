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
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Optional

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

def _run_faceage(image_paths: list[str]) -> FaceAgeResult:
    """
    Run FaceAge in batch mode across all face images.
    Returns aggregated FaceAgeResult (median biological age).
    """
    if not image_paths:
        return FaceAgeResult(status="skipped", error="No face images available")

    try:
        # FaceAge batch mode: pass image_paths list, get list of UnifiedResult back
        raw = _call_module("faceage", {"image_paths": image_paths}, timeout=600)
    except Exception as exc:
        return FaceAgeResult(status="error", error=str(exc))

    # raw may be a list (batch) or a single dict
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
        return FaceAgeResult(status="error", error="所有图像均未能成功预测生物年龄")

    median_age = statistics.median(bio_ages)
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


def _run_facettd(image_paths: list[str], age: int) -> FacettdResult:
    """
    Run facettd on each face image, aggregate with median.
    """
    if not image_paths:
        return FacettdResult(status="skipped", error="No face images available")

    values: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_facettd_one, p, age): p for p in image_paths}
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result is not None:
                values.append(result)

    if not values:
        return FacettdResult(status="error", error="facettd 未返回有效结果")

    return FacettdResult(
        remaining_life_median=statistics.median(values),
        remaining_life_values=values,
        status="ok",
    )


# ── Disease modules (Parkinsons / lung_cancer / skin_disease) ─────────────────

def _run_audio_disease_module(
    module_id: str,
    audio_paths: list[str],
    prob_key: str = "probability",
    label_key: str = "label",
) -> DiseaseRiskResult:
    """
    Generic aggregator for audio-based disease detection.
    Runs module once per audio clip, aggregates mean probability + majority vote.
    """
    if not audio_paths:
        return DiseaseRiskResult(
            module_id=module_id,
            status="skipped",
            error="No audio segments available",
        )

    probs: list[float] = []
    labels: list[str] = []

    for audio_path in audio_paths:
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
) -> DiseaseRiskResult:
    """
    Generic aggregator for image-based disease detection (e.g. SkinDisease).
    """
    if not image_paths:
        return DiseaseRiskResult(
            module_id=module_id,
            status="skipped",
            error="No face images available",
        )

    probs: list[float] = []
    labels: list[str] = []

    for image_path in image_paths:
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

    Returns:
        SampleAnalysisResult with all module outputs filled in.
    """
    face_paths = sample.face_image_paths
    audio_paths = sample.audio_segment_paths
    age = sample.age_at_capture

    logger.info("[%s] 开始聚合分析（%d 张脸，%d 段音频，年龄 %d）",
                sample.subject_name, len(face_paths), len(audio_paths), age)

    # Run all modules in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        f_faceage = executor.submit(_run_faceage, face_paths)
        f_facettd = executor.submit(_run_facettd, face_paths, age)
        f_parkinson = executor.submit(
            _run_audio_disease_module, "parkinsons", audio_paths,
            "prob_parkinson", "label",
        )
        f_lung = executor.submit(
            _run_audio_disease_module, "lung_cancer", audio_paths,
            "probability", "label",
        )
        f_skin = (
            executor.submit(
                _run_image_disease_module, "skin_disease", face_paths,
                "probability", "label",
            )
            if include_skin_disease else None
        )

        faceage_result = f_faceage.result()
        facettd_result = f_facettd.result()
        parkinson_result = f_parkinson.result()
        lung_result = f_lung.result()
        skin_result = f_skin.result() if f_skin is not None else None

    # Fill age_delta in FaceAgeResult
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
