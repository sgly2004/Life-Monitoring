"""
Preprocess pipeline — video → deduplicated face images + speaker audio clips.

Pipeline per VideoSample:
  1. Run FaceTrack to detect time segments where the target face appears.
  2. Extract sampled frames from those segments using OpenCV.
  3. Deduplicate extracted frames via perceptual hashing (phash_threshold=8).
  4. Score + filter for frontal faces; keep top-N (default 20).
  5. Extract full video audio with video_to_audio.extract_audio.
  6. Run VoiceTrack to detect time segments attributed to target speaker.
  7. Limit audio to max_audio_minutes (default 10 min) before cutting.
  8. Cut each speaker segment into a separate WAV file.
  9. Persist everything under uploads/<subject_name>/<YYYYMMDD_HHMMSS>/.

Progress callbacks:
  progress_cb(event_type: str, **kwargs) — called throughout pipeline.
  event types: "phase", "step"
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.module_runner import _load_module_python_exes, SCRIPT_MAP
from core.workflow_types import AnchorInput, PreprocessedSample, VideoSample
from utils.video_to_audio import extract_audio
from utils.audio_to_wav import to_wav

logger = logging.getLogger(__name__)

UPLOADS_DIR = ROOT_DIR / "uploads"

# ── Limits ────────────────────────────────────────────────────────────────────
DEFAULT_MAX_FACE_IMAGES = 20      # after dedup + frontal filtering
DEFAULT_PHASH_THRESHOLD = 8       # Hamming distance (0–64); lower = stricter
DEFAULT_MAX_AUDIO_MINUTES = 10.0  # clip voice segments at this total duration

# ── Perceptual hash dedup ────────────────────────────────────────────────────

def _phash_distance(img_path_a: Path, img_path_b: Path) -> int:
    """
    Compute perceptual hash distance between two images using PIL.
    Returns Hamming distance (0 = identical, 64 = completely different).
    Falls back to OpenCV mean-pixel diff if imagehash is unavailable.
    """
    try:
        import imagehash
        from PIL import Image
        h_a = imagehash.phash(Image.open(img_path_a))
        h_b = imagehash.phash(Image.open(img_path_b))
        return int(h_a - h_b)
    except ImportError:
        pass
    try:
        a = cv2.resize(cv2.imread(str(img_path_a), cv2.IMREAD_GRAYSCALE), (8, 8))
        b = cv2.resize(cv2.imread(str(img_path_b), cv2.IMREAD_GRAYSCALE), (8, 8))
        if a is None or b is None:
            return 0
        return abs(int(a.mean()) - int(b.mean()))
    except Exception:
        return 0


def _deduplicate_images(
    image_paths: list[Path],
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> list[Path]:
    """
    Remove near-duplicate images using perceptual hash comparison.
    Keeps the first image in each cluster.

    Args:
        image_paths:      Ordered list of candidate images.
        phash_threshold:  Max Hamming distance to consider "duplicate" (default 8).
                          Range 0–64; lower = stricter.
    """
    if not image_paths:
        return []

    kept: list[Path] = [image_paths[0]]
    for candidate in image_paths[1:]:
        is_dup = any(
            _phash_distance(candidate, ref) <= phash_threshold
            for ref in kept
        )
        if not is_dup:
            kept.append(candidate)

    return kept


# ── Frontal face scoring ─────────────────────────────────────────────────────

def _frontal_face_score(image_path: Path) -> float:
    """
    Score an image for frontal-face quality using OpenCV Haar cascade.
    Returns a float in [0, 1]:
        1.0  — a frontal face is clearly detected
        0.5  — face detected but not confidently frontal
        0.0  — no frontal face detected
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0.5

        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return 0.0

        areas = [fw * fh for (_, _, fw, fh) in faces]
        max_area = max(areas)
        score = min(max_area / (w * h), 1.0) * 2
        return min(score, 1.0)

    except Exception as exc:
        logger.debug("Frontal face scoring failed for %s: %s", image_path, exc)
        return 0.5


def _select_top_faces(
    image_paths: list[Path],
    max_images: int = DEFAULT_MAX_FACE_IMAGES,
) -> list[Path]:
    """
    Score images for frontal face quality and return the top-N.
    Preserves temporal order among the selected images.
    """
    if len(image_paths) <= max_images:
        return image_paths

    scored = [(p, _frontal_face_score(p)) for p in image_paths]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_paths = {p for p, _ in scored[:max_images]}
    return [p for p in image_paths if p in top_paths]


# ── Face crop from a single frame ────────────────────────────────────────────

def _crop_face_from_frame(frame) -> Optional[object]:
    """
    Detect and crop the largest face from a BGR frame (numpy array).

    Strategy:
      1. Try frontal Haar cascade (scaleFactor 1.1, minNeighbors 4).
      2. If none found, try profile Haar cascade (scaleFactor 1.1, minNeighbors 3).
      3. If still none, return None (caller decides what to do with the full frame).

    The crop includes 40 % padding around the detected bounding box so the
    models have enough facial context.

    Returns:
        Cropped BGR image (numpy array) or None if no face detected.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_total, w_total = frame.shape[:2]

        frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                         minSize=(40, 40))

        if len(faces) == 0:
            profile = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )
            faces = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,
                                             minSize=(40, 40))

        if len(faces) == 0:
            return None

        # Largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # 40 % padding for model context
        pad = int(max(w, h) * 0.4)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_total, x + w + pad)
        y2 = min(h_total, y + h + pad)

        return frame[y1:y2, x1:x2]
    except Exception:
        return None


# ── Frame extraction from video segments ─────────────────────────────────────

def _extract_frames_from_segments(
    video_path: Path,
    segments: list[dict],
    output_dir: Path,
    frames_per_second: float = 1.0,
) -> list[Path]:
    """
    Extract **face crops** from video segments using OpenCV + Haar cascade.

    For each sampled frame:
      - Attempt to detect and crop the face region (with 40 % padding).
      - If a face is detected, save the crop.
      - If no face is detected in a frame, skip it.
      - Fallback: if zero face crops were obtained across ALL frames, save
        the full frames instead (rare edge case).

    Args:
        video_path:        Input video file.
        segments:          List of {start_time, end_time, ...} dicts from FaceTrack.
        output_dir:        Where to save extracted face-crop JPEGs.
        frames_per_second: How many frames to sample per second of segment.

    Returns:
        List of saved face-crop image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(video_fps / frames_per_second))

    saved_paths: list[Path] = []
    fallback_frames: list[tuple[Path, object]] = []  # (path, frame) if no face found
    frame_counter = 0

    for seg in segments:
        start_s = float(seg.get("start_time", 0))
        end_s = float(seg.get("end_time", start_s + 1))
        start_frame = int(start_s * video_fps)
        end_frame = int(end_s * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % frame_step == 0:
                ts_ms = int(current_frame * 1000 / video_fps)
                out_path = output_dir / f"face_{frame_counter:05d}_{ts_ms}ms.jpg"

                crop = _crop_face_from_frame(frame)
                if crop is not None:
                    cv2.imwrite(str(out_path), crop)
                    saved_paths.append(out_path)
                else:
                    # Keep full frame as fallback candidate (not yet written)
                    fallback_frames.append((out_path, frame))

                frame_counter += 1

            current_frame += 1

    cap.release()

    # If we got no face crops at all, fall back to saving full frames
    if not saved_paths and fallback_frames:
        logger.warning(
            "未能从片段中检测到人脸，回退到保存 %d 帧完整画面", len(fallback_frames)
        )
        for path, frame in fallback_frames:
            cv2.imwrite(str(path), frame)
            saved_paths.append(path)

    return saved_paths


# ── FaceTrack subprocess call ─────────────────────────────────────────────────

def _run_facetrack(
    video_path: Path,
    anchor_face_path: Path,
    threshold: float = 0.70,
) -> list[dict]:
    """
    Call FaceTrack via subprocess, return list of clip dicts.
    Each clip has: start_time, end_time, duration, confidence.
    """
    script = SCRIPT_MAP["facetrack"]
    exe = _load_module_python_exes().get("facetrack") or sys.executable

    inputs = {
        "video_path": str(video_path),
        "reference_face_path": str(anchor_face_path),
        "threshold": threshold,
    }

    proc = subprocess.run(
        [exe, str(script)],
        input=json.dumps(inputs, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
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

    result = json.loads(json_line)
    if result.get("status") != "success":
        raise RuntimeError(f"FaceTrack 失败：{result.get('error') or result.get('summary')}")

    return result.get("outputs", {}).get("clips", [])


# ── VoiceTrack subprocess call ────────────────────────────────────────────────

def _run_voicetrack(
    audio_path: Path,
    anchor_voice_path: Path,
    threshold: float = 0.75,
) -> list[dict]:
    """
    Call VoiceTrack via subprocess, return list of clip dicts.
    Each clip has: start_time, end_time, duration, confidence.
    """
    script = SCRIPT_MAP["voicetrack"]
    exe = _load_module_python_exes().get("voicetrack") or sys.executable

    inputs = {
        "audio_path": str(audio_path),
        "reference_audio_path": str(anchor_voice_path),
        "threshold": threshold,
    }

    proc = subprocess.run(
        [exe, str(script)],
        input=json.dumps(inputs, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
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

    result = json.loads(json_line)
    if result.get("status") != "success":
        raise RuntimeError(f"VoiceTrack 失败：{result.get('error') or result.get('summary')}")

    return result.get("outputs", {}).get("clips", [])


# ── Audio segment extraction ──────────────────────────────────────────────────

def _extract_audio_segments(
    full_audio_path: Path,
    clips: list[dict],
    output_dir: Path,
    sample_rate: int = 44100,
) -> list[Path]:
    """
    Cut each VoiceTrack clip into a separate WAV file.
    Returns list of WAV paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths: list[Path] = []

    for i, clip in enumerate(clips):
        start_s = float(clip.get("start_time", 0))
        end_s = float(clip.get("end_time", start_s + 1))
        out_path = output_dir / f"voice_seg_{i:03d}_{int(start_s)}_{int(end_s)}.wav"
        try:
            extract_audio(
                video_path=full_audio_path,
                start=start_s,
                end=end_s,
                output_path=out_path,
                sample_rate=sample_rate,
                quiet=True,
            )
            wav_paths.append(out_path)
        except Exception as exc:
            logger.warning("音频片段提取失败 (clip %d): %s", i, exc)

    return wav_paths


def _limit_clips_by_duration(
    clips: list[dict],
    max_minutes: float,
) -> tuple[list[dict], float]:
    """
    Keep only enough clips to stay within max_minutes total duration.
    Returns (limited_clips, total_minutes_kept).
    """
    max_seconds = max_minutes * 60.0
    limited: list[dict] = []
    total = 0.0
    for clip in clips:
        dur = float(clip.get("end_time", 0)) - float(clip.get("start_time", 0))
        if total + dur > max_seconds and limited:
            break
        limited.append(clip)
        total += dur
    return limited, total / 60.0


# ── Main entry point ──────────────────────────────────────────────────────────

def preprocess_sample(
    anchor: AnchorInput,
    sample: VideoSample,
    uploads_root: Optional[Path] = None,
    face_threshold: float = 0.70,
    voice_threshold: float = 0.75,
    frames_per_second: float = 1.0,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    max_face_images: int = DEFAULT_MAX_FACE_IMAGES,
    max_audio_minutes: float = DEFAULT_MAX_AUDIO_MINUTES,
    progress_cb: Optional[Callable] = None,
) -> PreprocessedSample:
    """
    Full preprocessing pipeline for one video sample.

    Directory layout created on disk:
        uploads/<subject_name>/<YYYYMMDD_HHMMSS>/
            faces/           raw frames from FaceTrack segments
            faces_deduped/   deduplicated + frontal-filtered frames (≤ max_face_images)
            audio_full.wav   full video audio
            audio_segs/      per-speaker WAV segments (≤ max_audio_minutes total)

    Dedup parameters:
        phash_threshold = 8  (Hamming distance, range 0–64; lower = stricter)
        max_face_images = 20 (after dedup + frontal filtering)
        max_audio_minutes = 10 (total speaker audio kept)

    Args:
        anchor:             Anchor identity (name, face image, voice clip).
        sample:             Video sample with path, date, and age.
        uploads_root:       Root directory for all uploads (defaults to repo/uploads/).
        face_threshold:     Cosine similarity threshold for FaceTrack.
        voice_threshold:    Cosine similarity threshold for VoiceTrack.
        frames_per_second:  Frames to extract per second of matched face segment.
        phash_threshold:    Max perceptual hash Hamming distance for dedup (default 8).
        max_face_images:    Maximum deduplicated face images to retain (default 20).
        max_audio_minutes:  Maximum total speaker audio to retain in minutes (default 10).
        progress_cb:        Optional callable(event_type: str, **data) for progress events.

    Returns:
        PreprocessedSample with all paths filled in.
    """
    def _cb(event_type: str, **data):
        if progress_cb:
            try:
                progress_cb(event_type, **data)
            except Exception:
                pass

    if uploads_root is None:
        uploads_root = UPLOADS_DIR

    ts_str = sample.captured_at.strftime("%Y%m%d_%H%M%S")
    preprocess_dir = uploads_root / anchor.subject_name / ts_str
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(sample.video_path)
    anchor_face = Path(anchor.anchor_face_path)
    anchor_voice = Path(anchor.anchor_voice_path)

    # ── Step 1: FaceTrack — detect face segments ──────────────────────────────
    _cb("step", step="facetrack", desc="人脸追踪中…（FaceTrack）")
    logger.info("[%s] 运行 FaceTrack …", anchor.subject_name)
    try:
        face_clips = _run_facetrack(video_path, anchor_face, threshold=face_threshold)
        logger.info("[%s] FaceTrack 检测到 %d 个片段", anchor.subject_name, len(face_clips))
        _cb("step", step="facetrack_done",
            clips=len(face_clips),
            desc=f"✓ 人脸追踪完成，检测到 {len(face_clips)} 个面部片段")
    except Exception as exc:
        logger.warning("[%s] FaceTrack 失败，跳过人脸图像提取：%s", anchor.subject_name, exc)
        _cb("step", step="facetrack_done", clips=0, desc=f"⚠ FaceTrack 失败：{exc}")
        face_clips = []

    # ── Step 2: Extract frames from detected segments ─────────────────────────
    face_dir_raw = preprocess_dir / "faces"
    raw_frames: list[Path] = []
    if face_clips:
        _cb("step", step="frames", desc="从面部片段中抽取视频帧…")
        try:
            raw_frames = _extract_frames_from_segments(
                video_path=video_path,
                segments=face_clips,
                output_dir=face_dir_raw,
                frames_per_second=frames_per_second,
            )
            logger.info("[%s] 抽帧完成，共 %d 帧", anchor.subject_name, len(raw_frames))
        except Exception as exc:
            logger.warning("[%s] 抽帧失败：%s", anchor.subject_name, exc)

    # ── Step 3: Deduplicate frames ────────────────────────────────────────────
    face_dir_deduped = preprocess_dir / "faces_deduped"
    face_dir_deduped.mkdir(parents=True, exist_ok=True)

    deduped_frames = _deduplicate_images(raw_frames, phash_threshold=phash_threshold)
    logger.info(
        "[%s] 去重后 %d 帧（原 %d 帧）",
        anchor.subject_name, len(deduped_frames), len(raw_frames),
    )

    # ── Step 4: Filter for frontal faces and keep top-N ──────────────────────
    selected_frames = _select_top_faces(deduped_frames, max_images=max_face_images)
    logger.info("[%s] 正脸筛选后 %d 帧", anchor.subject_name, len(selected_frames))
    _cb("step", step="frames_done",
        raw=len(raw_frames),
        deduped=len(deduped_frames),
        selected=len(selected_frames),
        phash_threshold=phash_threshold,
        desc=(
            f"✓ 抽帧完成：原始 {len(raw_frames)} 帧 → "
            f"去重后 {len(deduped_frames)} 帧（phash≤{phash_threshold}）→ "
            f"正脸筛选保留 {len(selected_frames)} 张"
        ))

    import shutil as _shutil
    final_face_paths: list[str] = []
    for i, src in enumerate(selected_frames):
        dst = face_dir_deduped / f"selected_{i:03d}{src.suffix}"
        _shutil.copy2(src, dst)
        final_face_paths.append(str(dst))

    # ── Step 5: Extract full audio from video ────────────────────────────────
    _cb("step", step="audio_extract", desc="从视频中提取音频…")
    logger.info("[%s] 提取视频音频 …", anchor.subject_name)
    full_audio_path = preprocess_dir / "audio_full.wav"
    audio_ok = False
    try:
        extract_audio(
            video_path=video_path,
            output_path=full_audio_path,
            quiet=True,
        )
        audio_ok = True
        logger.info("[%s] 音频提取完成：%s", anchor.subject_name, full_audio_path)
        _cb("step", step="audio_extract_done", desc="✓ 音频提取完成")
    except Exception as exc:
        logger.warning("[%s] 音频提取失败：%s", anchor.subject_name, exc)
        _cb("step", step="audio_extract_done", desc=f"⚠ 音频提取失败：{exc}")

    # ── Step 6: VoiceTrack — detect speaker segments ──────────────────────────
    voice_seg_paths: list[str] = []
    if audio_ok:
        _cb("step", step="voicetrack", desc="声纹追踪中…（VoiceTrack）")
        logger.info("[%s] 运行 VoiceTrack …", anchor.subject_name)
        try:
            voice_clips = _run_voicetrack(
                full_audio_path, anchor_voice, threshold=voice_threshold
            )
            logger.info("[%s] VoiceTrack 检测到 %d 个说话片段", anchor.subject_name, len(voice_clips))
        except Exception as exc:
            logger.warning("[%s] VoiceTrack 失败：%s", anchor.subject_name, exc)
            _cb("step", step="voicetrack_done", clips=0, total_minutes=0.0,
                desc=f"⚠ VoiceTrack 失败：{exc}")
            voice_clips = []

        if voice_clips:
            # ── Step 6b: Limit to max_audio_minutes ──────────────────────────
            limited_clips, kept_minutes = _limit_clips_by_duration(
                voice_clips, max_audio_minutes
            )
            dropped = len(voice_clips) - len(limited_clips)
            logger.info(
                "[%s] 音频限制：保留 %d/%d 段（%.1f min）",
                anchor.subject_name, len(limited_clips), len(voice_clips), kept_minutes,
            )
            _cb("step", step="voicetrack_done",
                clips=len(limited_clips),
                total_clips=len(voice_clips),
                total_minutes=round(kept_minutes, 1),
                dropped=dropped,
                desc=(
                    f"✓ 声纹追踪完成：{len(limited_clips)}/{len(voice_clips)} 段"
                    f"（{kept_minutes:.1f} 分钟，限制 {max_audio_minutes} 分钟）"
                ))
            voice_clips = limited_clips

            # ── Step 7: Cut each speaker segment into WAV ─────────────────────
            _cb("step", step="audio_segs", desc=f"切割 {len(voice_clips)} 段说话音频…")
            audio_seg_dir = preprocess_dir / "audio_segs"
            seg_paths = _extract_audio_segments(
                full_audio_path=full_audio_path,
                clips=voice_clips,
                output_dir=audio_seg_dir,
            )
            voice_seg_paths = [str(p) for p in seg_paths]
            logger.info("[%s] 音频片段保存完成，共 %d 段", anchor.subject_name, len(voice_seg_paths))
            _cb("step", step="audio_segs_done",
                segments=len(voice_seg_paths),
                desc=f"✓ 音频切割完成，共 {len(voice_seg_paths)} 段")
        else:
            _cb("step", step="voicetrack_done", clips=0, total_minutes=0.0,
                desc="✓ 声纹追踪完成（未检测到目标说话人片段）")

    # ── Step 8: Write manifest for human inspection ───────────────────────────
    manifest = {
        "subject_name": anchor.subject_name,
        "captured_at": sample.captured_at.isoformat(),
        "age_at_capture": sample.age_at_capture,
        "video_path": str(video_path),
        "face_clips_count": len(face_clips),
        "raw_frames_count": len(raw_frames),
        "deduped_frames_count": len(deduped_frames),
        "selected_face_images": final_face_paths,
        "voice_segments": voice_seg_paths,
        "dedup_params": {
            "phash_threshold": phash_threshold,
            "max_face_images": max_face_images,
            "max_audio_minutes": max_audio_minutes,
        },
    }
    manifest_path = preprocess_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[%s] manifest 已写入：%s", anchor.subject_name, manifest_path)

    return PreprocessedSample(
        subject_name=anchor.subject_name,
        captured_at=sample.captured_at,
        age_at_capture=sample.age_at_capture,
        face_image_paths=final_face_paths,
        audio_segment_paths=voice_seg_paths,
        preprocess_dir=str(preprocess_dir),
    )
