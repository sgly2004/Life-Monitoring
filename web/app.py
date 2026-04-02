"""
Life Monitoring — FastAPI web backend

Endpoints:
  GET  /                        Serve index.html (综合分析)
  GET  /preprocessing           Serve preprocessing.html (数据预处理)
  GET  /age-timeline            Serve age_timeline.html
  GET  /lifetime-curve          Serve lifetime_curve.html (寿命曲线 Demo)
  GET  /api/modules             List all module metadata
  POST /api/analyze             Run analysis (multipart form)
  POST /api/faceage-batch       Batch FaceAge prediction (multiple images)
  GET  /api/config/template     Get current prompt template
  PUT  /api/config/template     Save updated prompt template
  POST /api/split/face          FaceTrack — split video by reference face
  POST /api/split/voice         VoiceTrack — split audio by reference voice
  POST /api/face-split          FaceTrack (frontend-facing, field: reference_image)
  POST /api/voice-split         VoiceTrack (frontend-facing, field: reference_audio)
  POST /api/workflow/analyze    Full lifetime-curve pipeline (anchor + dated videos)

Start:
  uvicorn web.app:app --reload --port 8000
"""
from __future__ import annotations
import concurrent.futures
import json as _json
import os
import sys
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.orchestrator import orchestrate, load_prompt_template, save_prompt_template

app = FastAPI(title="Life Monitoring", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
UPLOADS_DIR = ROOT_DIR / "uploads"
MODULES_CONFIG = ROOT_DIR / "config" / "modules.yaml"
UPLOADS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/preprocessing", include_in_schema=False)
async def preprocessing():
    return FileResponse(str(STATIC_DIR / "preprocessing.html"))


@app.get("/age-timeline", include_in_schema=False)
async def age_timeline():
    return FileResponse(str(STATIC_DIR / "age_timeline.html"))


@app.get("/lifetime-curve", include_in_schema=False)
async def lifetime_curve():
    return FileResponse(str(STATIC_DIR / "lifetime_curve.html"))


@app.get("/api/modules")
async def get_modules():
    """Return module registry from config/modules.yaml."""
    with open(MODULES_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("modules", [])


@app.get("/api/config/template")
async def get_template():
    """Return the current prompt template string."""
    return {"template": load_prompt_template()}


class TemplateUpdate(BaseModel):
    template: str


@app.put("/api/config/template")
async def update_template(body: TemplateUpdate):
    """Persist an updated prompt template."""
    try:
        save_prompt_template(body.template)
        return {"status": "saved"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/analyze")
async def analyze(
    age: int = Form(...),
    enabled_modules: str = Form(...),   # JSON array string, e.g. '["faceage","parkinsons"]'
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    prompt_override: Optional[str] = Form(None),
):
    """
    Run the enabled modules against the uploaded files.

    Form fields:
      age              int
      enabled_modules  JSON-encoded list of module IDs
      image            optional face image file
      audio            optional audio file (WAV/M4A)
      prompt_override  optional Jinja2 template string to use instead of saved one
    """
    import json as _json

    try:
        modules: List[str] = _json.loads(enabled_modules)
    except Exception:
        raise HTTPException(status_code=422, detail="enabled_modules must be a JSON array")

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    try:
        if image and image.filename:
            suffix = Path(image.filename).suffix or ".jpg"
            img_file = session_dir / f"face{suffix}"
            with open(img_file, "wb") as f:
                shutil.copyfileobj(image.file, f)
            image_path = str(img_file)

        if audio and audio.filename:
            suffix = Path(audio.filename).suffix or ".wav"
            aud_file = session_dir / f"audio{suffix}"
            with open(aud_file, "wb") as f:
                shutil.copyfileobj(audio.file, f)
            audio_path = str(aud_file)

        result = orchestrate(
            enabled_modules=modules,
            image_path=image_path,
            audio_path=audio_path,
            age=age,
            prompt_override=prompt_override if prompt_override else None,
        )

        return JSONResponse(content=result)

    finally:
        # Clean up uploaded files after response
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


@app.post("/api/faceage-batch")
async def faceage_batch(files: List[UploadFile] = File(...)):
    """
    Batch FaceAge prediction for multiple face images.

    Accepts any number of image files (e.g. from a folder upload).
    Runs all images in a single subprocess call (model loaded once).

    Returns list of:
      { filename, status, biological_age, detection_confidence, error }
    sorted by natural filename order.
    """
    import re
    import subprocess

    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

    try:
        saved: list[tuple[str, Path]] = []   # (original_filename, saved_path)
        for upload in files:
            if not upload.filename:
                continue
            ext = Path(upload.filename).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            # Use original filename to preserve sort order
            safe_name = upload.filename.replace("/", "_").replace("\\", "_")
            dest = session_dir / safe_name
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved.append((upload.filename, dest))

        if not saved:
            raise HTTPException(status_code=422, detail="No valid image files found")

        # Natural sort: numbers inside filenames treated numerically
        def _natural_key(item):
            parts = re.split(r"(\d+)", Path(item[0]).stem)
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        saved.sort(key=_natural_key)

        image_paths = [str(p) for _, p in saved]
        original_names = [name for name, _ in saved]

        # Call FaceAge batch mode (model loaded once for all images)
        from core.module_runner import _load_module_python_exes, ROOT_DIR
        faceage_script = ROOT_DIR / "module" / "FaceAge" / "run.py"
        exe = _load_module_python_exes().get("faceage") or sys.executable

        proc = subprocess.run(
            [exe, str(faceage_script)],
            input=_json.dumps({"image_paths": image_paths}, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
        )

        # Parse the JSON array from stdout (skip debug lines)
        json_line = None
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                json_line = line
                break

        if json_line is None:
            stderr_tail = proc.stderr[-1500:] if proc.stderr else "(no stderr)"
            raise RuntimeError(f"FaceAge 无输出。\nstderr:\n{stderr_tail}")

        raw_results: list[dict] = _json.loads(json_line)

        # Attach original filenames and flatten for the frontend
        output = []
        for orig_name, r in zip(original_names, raw_results):
            output.append(
                {
                    "filename": orig_name,
                    "status": r.get("status", "error"),
                    "biological_age": r.get("outputs", {}).get("biological_age"),
                    "detection_confidence": r.get("outputs", {}).get("detection_confidence"),
                    "summary": r.get("summary", ""),
                    "error": r.get("error"),
                }
            )

        return JSONResponse(content={"results": output})

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


# ── Split Tool Endpoints ─────────────────────────────────────────────────────

@app.post("/api/split/face")
async def split_by_face(
    video: UploadFile = File(...),
    reference_face: UploadFile = File(...),
    threshold: float = Form(0.70),
):
    """
    FaceTrack — split a video by a reference face.

    Form fields:
      video            face video file (MP4/AVI/MOV)
      reference_face   reference face image (JPG/PNG)
      threshold        cosine similarity threshold (default 0.70)
    """
    import subprocess

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded files
        video_ext = Path(video.filename).suffix.lower() if video.filename else ".mp4"
        video_file = session_dir / f"video{video_ext}"
        with open(video_file, "wb") as f:
            shutil.copyfileobj(video.file, f)

        ref_ext = Path(reference_face.filename).suffix.lower() if reference_face.filename else ".jpg"
        ref_file = session_dir / f"reference{ref_ext}"
        with open(ref_file, "wb") as f:
            shutil.copyfileobj(reference_face.file, f)

        # Prepare inputs
        inputs = {
            "video_path": str(video_file),
            "reference_face_path": str(ref_file),
            "threshold": float(threshold),
        }

        # Call FaceTrack module directly
        from core.module_runner import _load_module_python_exes, ROOT_DIR as _root
        facetrack_script = _root / "module" / "FaceTrack" / "run.py"
        exe = _load_module_python_exes().get("facetrack") or sys.executable

        proc = subprocess.run(
            [exe, str(facetrack_script)],
            input=_json.dumps(inputs, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": str(_root)},
        )

        # Parse result
        stdout = proc.stdout.strip()
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(empty)"
            raise RuntimeError(f"FaceTrack 无输出。\nstderr:\n{stderr_tail}")

        result: dict = _json.loads(json_line)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


@app.post("/api/split/voice")
async def split_by_voice(
    audio: UploadFile = File(...),
    reference_audio: UploadFile = File(...),
    threshold: float = Form(0.75),
):
    """
    VoiceTrack — split an audio by a reference voice.

    Form fields:
      audio             full audio file (WAV/MP3/M4A)
      reference_audio   reference voice sample (WAV/MP3/M4A)
      threshold         cosine similarity threshold (default 0.75)
    """
    import subprocess

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded files
        audio_ext = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
        audio_file = session_dir / f"audio{audio_ext}"
        with open(audio_file, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        ref_ext = Path(reference_audio.filename).suffix.lower() if reference_audio.filename else ".wav"
        ref_file = session_dir / f"reference{ref_ext}"
        with open(ref_file, "wb") as f:
            shutil.copyfileobj(reference_audio.file, f)

        # Prepare inputs
        inputs = {
            "audio_path": str(audio_file),
            "reference_audio_path": str(ref_file),
            "threshold": float(threshold),
        }

        # Call VoiceTrack module directly
        from core.module_runner import _load_module_python_exes, ROOT_DIR as _root
        voicetrack_script = _root / "module" / "VoiceTrack" / "run.py"
        exe = _load_module_python_exes().get("voicetrack") or sys.executable

        proc = subprocess.run(
            [exe, str(voicetrack_script)],
            input=_json.dumps(inputs, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": str(_root)},
        )

        # Parse result
        stdout = proc.stdout.strip()
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(empty)"
            raise RuntimeError(f"VoiceTrack 无输出。\nstderr:\n{stderr_tail}")

        result: dict = _json.loads(json_line)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


# ── Workflow: list existing subjects ─────────────────────────────────────────

@app.get("/api/workflow/subjects")
async def list_subjects():
    """
    Return a list of subject names that already have saved uploads.
    Scans uploads/<subject_name>/ directories for a manifest.json to confirm
    at least one completed preprocessing run exists.
    """
    subjects = []
    if UPLOADS_DIR.exists():
        for entry in sorted(UPLOADS_DIR.iterdir()):
            if entry.is_dir() and not entry.name.startswith("__"):
                # Check if any timestamp sub-directory has a manifest
                has_data = any(
                    (sub / "manifest.json").exists()
                    for sub in entry.iterdir()
                    if sub.is_dir()
                )
                if has_data:
                    # Find anchor face/voice paths from the first manifest
                    anchor_face = None
                    anchor_voice = None
                    try:
                        manifests = sorted(
                            sub / "manifest.json"
                            for sub in entry.iterdir()
                            if sub.is_dir() and (sub / "manifest.json").exists()
                        )
                        if manifests:
                            import json as _json
                            m = _json.loads(manifests[0].read_text(encoding="utf-8"))
                            # Anchors aren't in manifest — just report name + count
                    except Exception:
                        pass
                    subjects.append({
                        "subject_name": entry.name,
                        "sample_count": sum(
                            1 for sub in entry.iterdir()
                            if sub.is_dir() and (sub / "manifest.json").exists()
                        ),
                    })
    return JSONResponse(content={"subjects": subjects})


# ── Workflow: load persisted timeline ────────────────────────────────────────

@app.get("/api/workflow/timeline/{subject_name}")
async def get_timeline(subject_name: str):
    """
    Return the saved LifetimeTimeline for a subject, if it exists.
    The timeline is stored at uploads/<subject_name>/timeline.json and is
    updated every time a successful /api/workflow/analyze completes.
    """
    timeline_path = UPLOADS_DIR / subject_name / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="该人物尚无历史时间轴数据")
    try:
        import json as _json
        data = _json.loads(timeline_path.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Workflow: full lifetime-curve pipeline ────────────────────────────────────

def _serialize_timeline(timeline) -> dict:
    """Convert LifetimeTimeline to a JSON-serializable dict (no raw_analysis)."""
    return {
        "subject_name": timeline.subject_name,
        "points": [
            {
                "captured_at": p.captured_at.isoformat(),
                "age_at_capture": p.age_at_capture,
                "remaining_life_estimate": p.remaining_life_estimate,
                "uncertainty_low": p.uncertainty_low,
                "uncertainty_high": p.uncertainty_high,
                "key_factors": p.key_factors,
                "summary": p.summary,
            }
            for p in timeline.sorted_points()
        ],
    }


def _merge_timeline_dicts(existing: dict, new_points: list[dict]) -> dict:
    """
    Merge new timeline points into an existing timeline dict.
    Points are keyed by their date (YYYY-MM-DD); newer runs overwrite same-day points.
    Result is sorted ascending by date.
    """
    by_date: dict[str, dict] = {}
    for pt in existing.get("points", []):
        day = pt["captured_at"][:10]
        by_date[day] = pt
    for pt in new_points:
        day = pt["captured_at"][:10]
        by_date[day] = pt
    merged = sorted(by_date.values(), key=lambda p: p["captured_at"])
    return {"subject_name": existing.get("subject_name", ""), "points": merged}


@app.post("/api/workflow/analyze")
async def workflow_analyze(
    subject_name: str = Form(...),
    anchor_face: UploadFile = File(...),
    anchor_voice: UploadFile = File(...),
    # birth_date replaces per-video age: one date for the whole person
    birth_date: Optional[str] = Form(None),   # ISO: YYYY-MM-DD
    # Repeated fields: one entry per video
    videos: List[UploadFile] = File(...),
    dates: List[str] = Form(...),    # capture date per video (YYYY-MM-DD)
    include_skin_disease: bool = Form(False),
    face_threshold: float = Form(0.70),
    voice_threshold: float = Form(0.75),
):
    """
    Full lifetime-curve pipeline.

    Form fields:
      subject_name         str   — unique name / ID for the person
      anchor_face          file  — reference face image (JPG/PNG)
      anchor_voice         file  — reference voice clip (WAV/MP3/M4A)
      birth_date           str   — date of birth (YYYY-MM-DD); age is auto-computed
      videos               files — one or more dated video files
      dates                strs  — capture date per video (YYYY-MM-DD)
      include_skin_disease bool  — whether to run SkinDisease module (default False)
      face_threshold       float — FaceTrack similarity threshold (default 0.70)
      voice_threshold      float — VoiceTrack similarity threshold (default 0.75)

    Returns LifetimeTimeline as JSON (merges with any previously saved timeline).
    """
    import json as _json
    from datetime import datetime as _dt, date as _date

    if len(videos) != len(dates):
        raise HTTPException(
            status_code=422,
            detail="videos and dates must have the same number of items",
        )

    # Parse birth_date
    parsed_birth_date: Optional[_date] = None
    if birth_date:
        try:
            parsed_birth_date = _date.fromisoformat(birth_date)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid birth_date format: '{birth_date}'. Use YYYY-MM-DD.",
            )

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / f"__session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save anchor files
        face_ext = Path(anchor_face.filename).suffix.lower() if anchor_face.filename else ".jpg"
        face_path = session_dir / f"anchor_face{face_ext}"
        with open(face_path, "wb") as f:
            shutil.copyfileobj(anchor_face.file, f)

        voice_ext = Path(anchor_voice.filename).suffix.lower() if anchor_voice.filename else ".wav"
        voice_path = session_dir / f"anchor_voice{voice_ext}"
        with open(voice_path, "wb") as f:
            shutil.copyfileobj(anchor_voice.file, f)

        from core.workflow_types import AnchorInput, VideoSample
        from core.preprocess_pipeline import preprocess_sample
        from core.result_aggregator import aggregate_sample
        from core.timeline_builder import build_timeline

        anchor = AnchorInput(
            subject_name=subject_name,
            birth_date=parsed_birth_date,
            anchor_face_path=str(face_path),
            anchor_voice_path=str(voice_path),
        )

        # Save video files and build VideoSample list
        video_samples: List[VideoSample] = []
        for i, (video_upload, date_str) in enumerate(zip(videos, dates)):
            vid_ext = Path(video_upload.filename).suffix.lower() if video_upload.filename else ".mp4"
            vid_path = session_dir / f"video_{i:02d}{vid_ext}"
            with open(vid_path, "wb") as f:
                shutil.copyfileobj(video_upload.file, f)

            try:
                captured_at = _dt.fromisoformat(date_str)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid date format for video {i}: '{date_str}'. Use YYYY-MM-DD.",
                )

            # Auto-compute age from birth_date; require at least one of them
            age_at_capture = anchor.compute_age(captured_at)
            if age_at_capture is None:
                raise HTTPException(
                    status_code=422,
                    detail="请提供出生日期（birth_date）以便自动计算拍摄时年龄",
                )

            video_samples.append(VideoSample(
                video_path=str(vid_path),
                captured_at=captured_at,
                age_at_capture=age_at_capture,
            ))

        # Run pipeline for each video (sequential to avoid OOM on large models)
        import logging as _logging
        analyses = []
        for vs in video_samples:
            try:
                preprocessed = preprocess_sample(
                    anchor=anchor,
                    sample=vs,
                    uploads_root=UPLOADS_DIR,
                    face_threshold=face_threshold,
                    voice_threshold=voice_threshold,
                )
                analysis = aggregate_sample(
                    preprocessed,
                    include_skin_disease=include_skin_disease,
                )
                analyses.append(analysis)
            except Exception as exc:
                _logging.getLogger(__name__).warning(
                    "Pipeline failed for video %s: %s", vs.video_path, exc
                )

        if not analyses:
            raise HTTPException(
                status_code=500,
                detail="所有视频处理均失败，无法生成时间轴",
            )

        timeline = build_timeline(subject_name=subject_name, analyses=analyses)
        new_timeline_dict = _serialize_timeline(timeline)

        # ── Persist timeline: merge with any existing saved timeline ──────────
        subject_dir = UPLOADS_DIR / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = subject_dir / "timeline.json"

        if timeline_path.exists():
            try:
                existing = _json.loads(timeline_path.read_text(encoding="utf-8"))
                merged = _merge_timeline_dicts(existing, new_timeline_dict["points"])
                merged["subject_name"] = subject_name
                timeline_path.write_text(
                    _json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                final_dict = merged
            except Exception:
                timeline_path.write_text(
                    _json.dumps(new_timeline_dict, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                final_dict = new_timeline_dict
        else:
            timeline_path.write_text(
                _json.dumps(new_timeline_dict, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            final_dict = new_timeline_dict

        return JSONResponse(content=final_dict)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up temporary session files (not the persisted subject uploads)
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)


# ── Frontend-facing split routes ──────────────────────────────────────────────
# These mirror /api/split/face and /api/split/voice but use the field names
# that the frontend (main.js) sends: reference_image/video and reference_audio/audio.

@app.post("/api/face-split")
async def face_split(
    video: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    threshold: float = Form(0.70),
):
    """
    Frontend-facing wrapper for FaceTrack.
    Forwards to /api/split/face after saving files.
    """
    import subprocess

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_ext = Path(video.filename).suffix.lower() if video.filename else ".mp4"
        video_file = session_dir / f"video{video_ext}"
        with open(video_file, "wb") as f:
            shutil.copyfileobj(video.file, f)

        ref_ext = Path(reference_image.filename).suffix.lower() if reference_image.filename else ".jpg"
        ref_file = session_dir / f"reference{ref_ext}"
        with open(ref_file, "wb") as f:
            shutil.copyfileobj(reference_image.file, f)

        inputs = {
            "video_path": str(video_file),
            "reference_face_path": str(ref_file),
            "threshold": float(threshold),
        }

        from core.module_runner import _load_module_python_exes, ROOT_DIR as _root
        facetrack_script = _root / "module" / "FaceTrack" / "run.py"
        exe = _load_module_python_exes().get("facetrack") or sys.executable

        proc = subprocess.run(
            [exe, str(facetrack_script)],
            input=_json.dumps(inputs, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": str(_root)},
        )

        stdout = proc.stdout.strip()
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(empty)"
            raise RuntimeError(f"FaceTrack 无输出。\nstderr:\n{stderr_tail}")

        raw: dict = _json.loads(json_line)
        # Normalize to the format the frontend expects
        return JSONResponse(content=_normalize_face_split_result(raw))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


@app.post("/api/voice-split")
async def voice_split(
    audio: UploadFile = File(...),
    reference_audio: UploadFile = File(...),
    threshold: float = Form(0.75),
):
    """
    Frontend-facing wrapper for VoiceTrack.
    Forwards to /api/split/voice after saving files.
    """
    import subprocess

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_ext = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
        audio_file = session_dir / f"audio{audio_ext}"
        with open(audio_file, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        ref_ext = Path(reference_audio.filename).suffix.lower() if reference_audio.filename else ".wav"
        ref_file = session_dir / f"reference{ref_ext}"
        with open(ref_file, "wb") as f:
            shutil.copyfileobj(reference_audio.file, f)

        inputs = {
            "audio_path": str(audio_file),
            "reference_audio_path": str(ref_file),
            "threshold": float(threshold),
        }

        from core.module_runner import _load_module_python_exes, ROOT_DIR as _root
        voicetrack_script = _root / "module" / "VoiceTrack" / "run.py"
        exe = _load_module_python_exes().get("voicetrack") or sys.executable

        proc = subprocess.run(
            [exe, str(voicetrack_script)],
            input=_json.dumps(inputs, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": str(_root)},
        )

        stdout = proc.stdout.strip()
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else "(empty)"
            raise RuntimeError(f"VoiceTrack 无输出。\nstderr:\n{stderr_tail}")

        raw: dict = _json.loads(json_line)
        return JSONResponse(content=_normalize_voice_split_result(raw))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


# ── Normalizers ─────────────────────────────────────────────────────────────

def _normalize_face_split_result(raw: dict) -> dict:
    """
    Convert FaceTrack's UnifiedResult schema to the flat format
    the frontend render function expects.
    """
    outputs = raw.get("outputs", {})
    clips: list[dict] = outputs.get("clips", [])

    segments = []
    matched_count = 0
    total_duration = 0.0

    for clip in clips:
        start = clip.get("start_time", 0)
        end = clip.get("end_time", 0)
        score = clip.get("confidence", 0.0)
        is_match = True  # FaceTrack only returns clips that already passed the threshold

        segments.append({
            "start_time": f"{start:.1f}s",
            "end_time":   f"{end:.1f}s",
            "score":      round(score, 4),
            "matched":    bool(is_match),
        })
        if is_match:
            matched_count += 1
            total_duration += (end - start)

    return {
        "total_segments":    len(clips),
        "matched_segments": matched_count,
        "total_duration":    f"{total_duration:.1f}s",
        "segments":          segments,
    }


def _normalize_voice_split_result(raw: dict) -> dict:
    """
    Convert VoiceTrack's UnifiedResult schema to the flat format
    the frontend render function expects.
    """
    outputs = raw.get("outputs", {})
    clips: list[dict] = outputs.get("clips", [])

    segments = []
    matched_count = 0
    total_duration = 0.0

    for clip in clips:
        start = clip.get("start_time", 0)
        end = clip.get("end_time", 0)
        score = clip.get("confidence", 0.0)
        is_match = True  # VoiceTrack only returns clips that already passed the threshold

        segments.append({
            "start_time": f"{start:.1f}s",
            "end_time":   f"{end:.1f}s",
            "score":      round(score, 4),
            "matched":    bool(is_match),
        })
        if is_match:
            matched_count += 1
            total_duration += (end - start)

    return {
        "total_segments":    len(clips),
        "matched_segments": matched_count,
        "total_duration":    f"{total_duration:.1f}s",
        "segments":          segments,
    }
