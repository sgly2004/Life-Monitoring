"""
Preprocess pipeline — video → deduplicated face images + speaker audio clips.

Pipeline per VideoSample:
  1. Run FaceTrack to detect time segments where the target face appears.
  2. Extract sampled frames from those segments using OpenCV.
  3. Deduplicate extracted frames via perceptual hashing.
  4. (Optional) Score + filter for frontal faces; keep top-N.
  5. Extract full video audio with video_to_audio.extract_audio.
  6. Run VoiceTrack to detect time segments attributed to target speaker.
  7. Cut each speaker segment into a separate WAV file.
  8. Persist everything under uploads/<subject_name>/<YYYYMMDD_HHMMSS>/.

All intermediate files are kept on disk so they can be inspected manually.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.module_runner import _load_module_python_exes, SCRIPT_MAP
from core.workflow_types import AnchorInput, PreprocessedSample, VideoSample
from utils.video_to_audio import extract_audio
from utils.audio_to_wav import to_wav

logger = logging.getLogger(__name__)

UPLOADS_DIR = ROOT_DIR / "uploads"

# Maximum face images to pass to health modules (configurable)
DEFAULT_MAX_FACE_IMAGES = 10

# ── Perceptual hash dedup ────────────────────────────────────────────────────

def _phash_distance(img_path_a: Path, img_path_b: Path) -> int:
    """
    Compute perceptual hash distance between two images using PIL.
    Returns Hamming distance (0 = identical, 64 = completely different).
    Falls back to 0 (treated as duplicate) if PIL / imagehash not available.
    """
    try:
        import imagehash
        from PIL import Image
        h_a = imagehash.phash(Image.open(img_path_a))
        h_b = imagehash.phash(Image.open(img_path_b))
        return int(h_a - h_b)
    except ImportError:
        # imagehash not installed — fall back to simple file-size heuristic
        pass
    # Fallback: compare mean pixel value of small thumbnail using OpenCV
    try:
        a = cv2.resize(cv2.imread(str(img_path_a), cv2.IMREAD_GRAYSCALE), (8, 8))
        b = cv2.resize(cv2.imread(str(img_path_b), cv2.IMREAD_GRAYSCALE), (8, 8))
        if a is None or b is None:
            return 0
        diff = abs(int(a.mean()) - int(b.mean()))
        return diff  # rough proxy; treat > 5 as different
    except Exception:
        return 0


def _deduplicate_images(
    image_paths: list[Path],
    phash_threshold: int = 8,
) -> list[Path]:
    """
    Remove near-duplicate images using perceptual hash comparison.
    Keeps the first image in each cluster.

    Args:
        image_paths:      Ordered list of candidate images.
        phash_threshold:  Max Hamming distance to consider "duplicate" (default 8).
                          Range 0–64; lower = stricter dedup.
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
    Uses zero extra dependencies beyond opencv-python which is already required
    by FaceTrack.
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0.5  # cascade not available — treat as neutral

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

        # Score by largest detected face area relative to image area
        areas = [fw * fh for (_, _, fw, fh) in faces]
        max_area = max(areas)
        score = min(max_area / (w * h), 1.0) * 2  # scale: half-image face → 1.0
        return min(score, 1.0)

    except Exception as exc:
        logger.debug("Frontal face scoring failed for %s: %s", image_path, exc)
        return 0.5  # unknown — treat as neutral


def _select_top_faces(
    image_paths: list[Path],
    max_images: int = DEFAULT_MAX_FACE_IMAGES,
) -> list[Path]:
    """
    Score images for frontal face quality and return the top-N.
    Falls back to simple front-of-list selection if scoring fails.
    """
    if len(image_paths) <= max_images:
        return image_paths

    scored = [(p, _frontal_face_score(p)) for p in image_paths]
    scored.sort(key=lambda x: x[1], reverse=True)
    # Re-sort by original position among top-N to preserve temporal order
    top_paths = {p for p, _ in scored[:max_images]}
    return [p for p in image_paths if p in top_paths]


# ── Frame extraction from video segments ─────────────────────────────────────

def _extract_frames_from_segments(
    video_path: Path,
    segments: list[dict],
    output_dir: Path,
    frames_per_second: float = 1.0,
) -> list[Path]:
    """
    Extract frames from video at specified time segments using OpenCV.

    Args:
        video_path:        Input video file.
        segments:          List of {start_time, end_time, ...} dicts from FaceTrack.
        output_dir:        Where to save extracted JPEGs.
        frames_per_second: How many frames to extract per second of segment.

    Returns:
        List of extracted image paths (unsorted).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(video_fps / frames_per_second))

    saved_paths: list[Path] = []
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
                out_path = output_dir / f"frame_{frame_counter:05d}_{ts_ms}ms.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_paths.append(out_path)
                frame_counter += 1

            current_frame += 1

    cap.release()
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


# ── Main entry point ──────────────────────────────────────────────────────────

def preprocess_sample(
    anchor: AnchorInput,
    sample: VideoSample,
    uploads_root: Optional[Path] = None,
    face_threshold: float = 0.70,
    voice_threshold: float = 0.75,
    frames_per_second: float = 1.0,
    phash_threshold: int = 8,
    max_face_images: int = DEFAULT_MAX_FACE_IMAGES,
) -> PreprocessedSample:
    """
    Full preprocessing pipeline for one video sample.

    Directory layout created on disk:
        uploads/<subject_name>/<YYYYMMDD_HHMMSS>/
            faces/           raw frames from FaceTrack segments
            faces_deduped/   deduplicated + frontal-filtered frames
            audio_full.wav   full video audio
            audio_segs/      per-speaker WAV segments from VoiceTrack

    Args:
        anchor:             Anchor identity (name, face image, voice clip).
        sample:             Video sample with path, date, and age.
        uploads_root:       Root directory for all uploads (defaults to repo/uploads/).
        face_threshold:     Cosine similarity threshold for FaceTrack.
        voice_threshold:    Cosine similarity threshold for VoiceTrack.
        frames_per_second:  Frames to extract per second of matched face segment.
        phash_threshold:    Max perceptual hash distance to treat images as duplicates.
        max_face_images:    Maximum deduplicated face images to retain.

    Returns:
        PreprocessedSample with all paths filled in.
    """
    if uploads_root is None:
        uploads_root = UPLOADS_DIR

    # Build persistent directory name from subject + timestamp
    ts_str = sample.captured_at.strftime("%Y%m%d_%H%M%S")
    preprocess_dir = uploads_root / anchor.subject_name / ts_str
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(sample.video_path)
    anchor_face = Path(anchor.anchor_face_path)
    anchor_voice = Path(anchor.anchor_voice_path)

    # ── Step 1: FaceTrack — detect face segments ──────────────────────────────
    logger.info("[%s] 运行 FaceTrack …", anchor.subject_name)
    try:
        face_clips = _run_facetrack(video_path, anchor_face, threshold=face_threshold)
        logger.info("[%s] FaceTrack 检测到 %d 个片段", anchor.subject_name, len(face_clips))
    except Exception as exc:
        logger.warning("[%s] FaceTrack 失败，跳过人脸图像提取：%s", anchor.subject_name, exc)
        face_clips = []

    # ── Step 2: Extract frames from detected segments ─────────────────────────
    face_dir_raw = preprocess_dir / "faces"
    raw_frames: list[Path] = []
    if face_clips:
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

    # Copy selected frames to deduped directory for easy inspection
    import shutil
    final_face_paths: list[str] = []
    for i, src in enumerate(selected_frames):
        dst = face_dir_deduped / f"selected_{i:03d}{src.suffix}"
        shutil.copy2(src, dst)
        final_face_paths.append(str(dst))

    # ── Step 5: Extract full audio from video ────────────────────────────────
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
    except Exception as exc:
        logger.warning("[%s] 音频提取失败：%s", anchor.subject_name, exc)

    # ── Step 6: VoiceTrack — detect speaker segments ──────────────────────────
    voice_seg_paths: list[str] = []
    if audio_ok:
        logger.info("[%s] 运行 VoiceTrack …", anchor.subject_name)
        try:
            voice_clips = _run_voicetrack(
                full_audio_path, anchor_voice, threshold=voice_threshold
            )
            logger.info("[%s] VoiceTrack 检测到 %d 个说话片段", anchor.subject_name, len(voice_clips))
        except Exception as exc:
            logger.warning("[%s] VoiceTrack 失败：%s", anchor.subject_name, exc)
            voice_clips = []

        # ── Step 7: Cut each speaker segment into WAV ─────────────────────────
        if voice_clips:
            audio_seg_dir = preprocess_dir / "audio_segs"
            seg_paths = _extract_audio_segments(
                full_audio_path=full_audio_path,
                clips=voice_clips,
                output_dir=audio_seg_dir,
            )
            voice_seg_paths = [str(p) for p in seg_paths]
            logger.info("[%s] 音频片段保存完成，共 %d 段", anchor.subject_name, len(voice_seg_paths))

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
