"""
VoiceTrack core — speaker diarization + voice embedding comparison.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from resemblyzer import VoiceEncoder
    _RESEMBLYZER_AVAILABLE = True
except ImportError:
    _RESEMBLYZER_AVAILABLE = False


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
    """Return best available device: cuda > mps (Apple Silicon) > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Voice Encoder (lazy singleton)
# ---------------------------------------------------------------------------

_voice_encoder: Optional[VoiceEncoder] = None


def _init_encoder():
    global _voice_encoder
    if _voice_encoder is not None:
        return
    if not _RESEMBLYZER_AVAILABLE:
        raise RuntimeError(
            "resemblyzer 未安装，请运行：\n"
            "  pip install resemblyzer\n"
        )
    _voice_encoder = VoiceEncoder()


# ---------------------------------------------------------------------------
# Reference voice embedding
# ---------------------------------------------------------------------------

def extract_voice_embedding(audio_path: str) -> np.ndarray:
    """
    Extract a voice embedding from a reference audio clip.

    Returns:
        1-D numpy array — normalised voice embedding vector.
        resemblyzer default dimension is 256.

    Raises:
        RuntimeError: if audio is too short or silent.
    """
    _init_encoder()
    from resemblyzer.audio import preprocess_wav

    try:
        wav = preprocess_wav(audio_path)
    except Exception as exc:
        raise RuntimeError(f"无法读取音频文件 {audio_path}：{exc}")

    if len(wav) < 16000:
        raise RuntimeError(
            f"参考音频过短（{len(wav)} 采样点 < 1秒），"
            "请提供至少 1 秒的有效语音"
        )

    emb = _voice_encoder.embed_utterance(wav)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb


# ---------------------------------------------------------------------------
# Speaker diarization + voice matching
# ---------------------------------------------------------------------------

def track_voice_in_audio(
    audio_path: str,
    reference_audio_path: str,
    threshold: float = 0.75,
) -> tuple[list[dict], float]:
    """
    Identify target speaker segments in a full audio file.

    Pipeline:
      1. Run speaker diarization (pyannote) → list of (start, end, speaker_id)
      2. Extract voice embedding for each segment
      3. Compare with reference embedding → cosine similarity
      4. Keep segments where similarity >= threshold
      5. Merge adjacent kept segments

    Args:
        audio_path:             Path to full audio file.
        reference_audio_path:   Path to reference audio of the target speaker.
        threshold:              Cosine similarity threshold (0.0–1.0).

    Returns:
        (clips, total_duration):
            clips: List of clip dicts with keys:
                   start_time, end_time, duration, confidence
            total_duration: Sum of all clip durations in seconds.
    """
    ref_emb = extract_voice_embedding(reference_audio_path)

    # Try pyannote for speaker diarization
    try:
        segments = _diarize_with_pyannote(audio_path)
    except Exception as exc:
        # Fallback: simple energy-based VAD then split into equal chunks
        segments = _fallback_segment(audio_path)

    # Compare each segment
    matched: list[dict] = []
    for seg_start, seg_end in segments:
        try:
            emb = _extract_segment_embedding(audio_path, seg_start, seg_end)
        except Exception:
            continue

        sim = _cosine_similarity(emb, ref_emb)
        if sim >= threshold:
            matched.append({
                "start_time": round(seg_start, 2),
                "end_time": round(seg_end, 2),
                "duration": round(seg_end - seg_start, 2),
                "confidence": round(float(sim), 3),
            })

    # Merge adjacent / overlapping segments
    clips = _merge_voice_clips(matched)
    total_duration = sum(c["duration"] for c in clips)

    return clips, total_duration


def _extract_segment_embedding(
    audio_path: str,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """
    Extract voice embedding for a segment of an audio file.
    """
    _init_encoder()
    from resemblyzer.audio import preprocess_wav
    import librosa

    # Load the full audio, extract segment
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)

    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    segment_wav = wav[start_sample:end_sample]

    if len(segment_wav) < 1600:  # less than 100ms
        raise RuntimeError("音频片段过短")

    emb = _voice_encoder.embed_utterance(segment_wav)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb


def _diarize_with_pyannote(audio_path: str) -> list[tuple[float, float]]:
    """
    Run pyannote-audio speaker diarization.
    Returns list of (start, end) tuples in seconds.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError(
            "pyannote.audio 未安装，请运行：\n"
            "  pip install pyannote.audio\n"
            "  # 并设置环境变量 HF_TOKEN（需要 Hugging Face token）"
        )

    device = _get_device()

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN", ""),
        )
    except Exception as exc:
        raise RuntimeError(
            f"pyannote pipeline 加载失败：{exc}\n"
            "请确保设置了环境变量 HF_TOKEN（需要 Hugging Face 账号的 access token）"
        )

    if device != "cpu":
        try:
            import torch
            pipeline.to(torch.device(device))
        except Exception:
            pass

    diarization = pipeline(audio_path)

    segments: list[tuple[float, float]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end))

    return segments


def _fallback_segment(audio_path: str) -> list[tuple[float, float]]:
    """
    Simple energy-based fallback when pyannote is unavailable.
    Splits audio into 5-second chunks.
    """
    import librosa

    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    chunk_s = 5.0
    chunk_samples = int(chunk_s * sr)

    segments: list[tuple[float, float]] = []
    for start in range(0, len(wav), chunk_samples):
        end_sample = min(start + chunk_samples, len(wav))
        chunk = wav[start:end_sample]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > 0.005:  # skip near-silent chunks
            segments.append((start / sr, end_sample / sr))

    return segments


def _merge_voice_clips(clips: list[dict]) -> list[dict]:
    """
    Merge adjacent / overlapping voice clips.
    Clips with gap <= MAX_GAP_S are merged.
    """
    if not clips:
        return []

    MAX_GAP_S = 1.5   # max gap to merge (seconds)
    MIN_DURATION_S = 0.5

    sorted_clips = sorted(clips, key=lambda x: x["start_time"])

    merged: list[dict] = []
    cur = dict(sorted_clips[0])

    for clip in sorted_clips[1:]:
        gap = clip["start_time"] - cur["end_time"]
        if gap <= MAX_GAP_S:
            # Merge
            new_end = clip["end_time"]
            dur = new_end - cur["start_time"]
            avg_conf = (
                cur["confidence"] * cur["duration"] +
                clip["confidence"] * clip["duration"]
            ) / dur
            cur["end_time"] = new_end
            cur["duration"] = round(dur, 2)
            cur["confidence"] = round(avg_conf, 3)
        else:
            if cur["duration"] >= MIN_DURATION_S:
                merged.append(cur)
            cur = dict(clip)

    if cur["duration"] >= MIN_DURATION_S:
        merged.append(cur)

    return merged


import os
