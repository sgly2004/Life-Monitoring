"""
FaceTracker — Video face tracking using facenet-pytorch embeddings.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Optional

# facenet-pytorch ships its model weights with the package — no manual download needed.
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _get_device() -> str:
    """
    Return best available device for CUDA, else 'cpu'.
    MPS (Apple Silicon) is intentionally excluded: both MTCNN and
    InceptionResnetV1 use adaptive_avg_pool2d which has a known PyTorch
    MPS bug for non-divisible input sizes (pytorch#96056).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Face detector + embedder (lazy singleton)
# ---------------------------------------------------------------------------

_mtcnn: Optional[MTCNN] = None
_embedder: Optional[InceptionResnetV1] = None
_device: str = "cpu"


_mtcnn_single: Optional[MTCNN] = None   # for reference image (single best face)
_mtcnn_multi:  Optional[MTCNN] = None   # for video frames (all faces)


def _init_models():
    global _mtcnn, _mtcnn_single, _mtcnn_multi, _embedder, _device
    if _embedder is not None:
        return
    if not _FACENET_AVAILABLE:
        raise RuntimeError(
            "facenet-pytorch 未安装，请运行：\n"
            "  pip install facenet-pytorch\n"
            "或使用 mediapipe 方案（见 face_tracker_mediapipe.py）"
        )
    _device = _get_device()

    common_kwargs = dict(
        image_size=160,
        margin=20,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=_device,
    )
    # keep_all=False → returns best single face (for reference images)
    _mtcnn_single = MTCNN(keep_all=False, **common_kwargs)
    # keep_all=True  → returns ALL faces in frame (for multi-person video)
    _mtcnn_multi  = MTCNN(keep_all=True,  **common_kwargs)
    # legacy alias
    _mtcnn = _mtcnn_single

    _embedder = InceptionResnetV1(pretrained="vggface2").eval().to(_device)


def extract_face_embedding(image_path: str) -> np.ndarray:
    """
    Extract a 512-dimensional face embedding from a reference image.

    Uses keep_all=False to pick the single best face (reference images
    are expected to contain exactly one person).

    Returns:
        1-D numpy array of shape (512,) — normalised face embedding vector.

    Raises:
        RuntimeError: if no face is detected.
    """
    import torch as _torch
    _init_models()

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Returns a single (3, 160, 160) tensor or None
    face = _mtcnn_single(img_rgb)
    if face is None:
        raise RuntimeError("参考图片中未检测到人脸，请上传清晰正面照")

    face_tensor = face.unsqueeze(0).to(_device)

    with _torch.no_grad():
        embedding: np.ndarray = _embedder(face_tensor).cpu().numpy().flatten()

    norm = np.linalg.norm(embedding)
    if norm > 1e-10:
        embedding /= norm

    return embedding


def track_faces_in_video(
    video_path: str,
    reference_face_path: str,
    threshold: float = 0.70,
    frame_interval: int = 5,
    min_face_px: int = 60,
) -> tuple[list[dict], float]:
    """
    Track target person in video by comparing face embeddings.

    Args:
        video_path:          Path to input video file.
        reference_face_path: Path to reference face image.
        threshold:           Cosine similarity threshold (0.0–1.0).
        frame_interval:      Process every N frames (skip for speed).

    Returns:
        (clips, total_duration):
            clips: List of clip dicts with keys:
                   start_time, end_time, duration, confidence
            total_duration: Sum of all clip durations in seconds.
    """
    import torch
    _init_models()

    ref_embedding = extract_face_embedding(reference_face_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    face_scores: list[tuple[int, float]] = []  # (frame_number, best_similarity)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # Detect bounding boxes to filter by size before embedding
                boxes, probs = _mtcnn_multi.detect(frame_rgb)
            except Exception:
                frame_idx += 1
                continue

            if boxes is not None:
                # Only keep faces large enough for reliable recognition
                large_boxes = [
                    b for b, p in zip(boxes, probs)
                    if p is not None and p > 0.7
                    and (b[2] - b[0]) >= min_face_px
                    and (b[3] - b[1]) >= min_face_px
                ]

                if not large_boxes:
                    frame_idx += 1
                    continue

                # Crop and embed only the large faces
                pil_frame = Image.fromarray(frame_rgb)
                face_crops = []
                for b in large_boxes:
                    x1, y1, x2, y2 = [int(v) for v in b]
                    margin = 10
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame_rgb.shape[1], x2 + margin)
                    y2 = min(frame_rgb.shape[0], y2 + margin)
                    crop = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))
                    face_crops.append(np.array(crop))

                # Stack into tensor: normalise to [-1, 1]
                crops_arr = np.stack(face_crops).astype(np.float32) / 127.5 - 1.0
                # (N, H, W, C) → (N, C, H, W)
                crops_t = torch.tensor(
                    crops_arr.transpose(0, 3, 1, 2), dtype=torch.float32
                ).to(_device)

                with torch.no_grad():
                    embeddings = _embedder(crops_t).cpu().numpy()  # (N, 512)

                # Normalise each embedding row
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                embeddings = embeddings / norms

                # Best similarity across large faces in this frame
                sims = [_cosine_similarity(e, ref_embedding) for e in embeddings]
                best_sim = float(max(sims))
                face_scores.append((frame_idx, best_sim))

        frame_idx += 1

    cap.release()

    clips = _merge_face_scores(face_scores, fps, threshold)
    total_duration = sum(c["duration"] for c in clips)

    return clips, total_duration


def _merge_face_scores(
    scores: list[tuple[int, float]],
    fps: float,
    threshold: float,
) -> list[dict]:
    """
    Convert per-frame similarity scores to merged continuous clips.

    Logic:
      - Frames with similarity >= threshold are marked as "target person"
      - Consecutive target frames within MAX_GAP frames are merged
      - Very short clips (< MIN_DURATION_S) are discarded
    """
    if not scores:
        return []

    MAX_GAP_S = 2.0          # max gap (seconds) between clips to merge
    MIN_DURATION_S = 1.0      # discard clips shorter than this
    MAX_GAP = int(MAX_GAP_S * fps)

    hit_frames = dict(scores)
    all_frame_nums = sorted(hit_frames.keys())
    is_hit = {f: (s >= threshold) for f, s in hit_frames.items()}

    segments: list[dict] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_confidences: list[float] = []

    for f in all_frame_nums:
        if not is_hit.get(f, False):
            continue

        sim = hit_frames[f]
        if cur_start is None:
            cur_start = f
            cur_end = f
            cur_confidences = [sim]
        elif f - cur_end <= MAX_GAP:
            cur_end = f
            cur_confidences.append(sim)
        else:
            dur = (cur_end - cur_start) / fps
            if dur >= MIN_DURATION_S:
                segments.append({
                    "start_time": round(cur_start / fps, 2),
                    "end_time":   round(cur_end   / fps, 2),
                    "duration":   round(dur, 2),
                    "confidence": round(float(np.mean(cur_confidences)), 3),
                })
            cur_start = f
            cur_end = f
            cur_confidences = [sim]

    if cur_start is not None:
        dur = (cur_end - cur_start) / fps
        if dur >= MIN_DURATION_S:
            segments.append({
                "start_time": round(cur_start / fps, 2),
                "end_time":   round(cur_end   / fps, 2),
                "duration":   round(dur, 2),
                "confidence": round(float(np.mean(cur_confidences)), 3),
            })

    return segments
