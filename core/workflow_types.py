"""
Workflow-level data types for the lifetime-curve pipeline.

Data flow:
  AnchorInput + VideoSample
    -> preprocess_pipeline  -> PreprocessedSample  (persisted to disk)
    -> result_aggregator    -> SampleAnalysisResult
    -> timeline_builder     -> LifetimeTimeline
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class AnchorInput(BaseModel):
    """Identity anchor for a single subject."""

    subject_name: str = Field(
        ...,
        description=(
            "Unique name / ID of the target person. "
            "Used as the top-level directory name for all persisted results, "
            "and as the key that associates multiple video samples into one timeline."
        ),
    )
    birth_date: Optional[date] = Field(
        None,
        description=(
            "Date of birth of the target person (ISO format: YYYY-MM-DD). "
            "When provided, age_at_capture for each VideoSample is automatically "
            "computed as floor((captured_at.date() - birth_date).days / 365.25). "
            "If not provided, VideoSample.age_at_capture must be supplied explicitly."
        ),
    )
    anchor_face_path: str = Field(
        ..., description="Path to a clear face image of the target person (JPG/PNG)."
    )
    anchor_voice_path: str = Field(
        ..., description="Path to a reference audio clip of the target person (WAV/MP3/M4A)."
    )

    def compute_age(self, captured_at: datetime) -> Optional[int]:
        """Return age in whole years at the given capture datetime, or None if birth_date is unset."""
        if self.birth_date is None:
            return None
        delta_days = (captured_at.date() - self.birth_date).days
        return max(0, int(delta_days / 365.25))


class VideoSample(BaseModel):
    """A single dated video sample for one subject."""

    video_path: str = Field(..., description="Path to the input video file (MP4/AVI/MOV).")
    captured_at: datetime = Field(
        ..., description="Date/time when the video was recorded."
    )
    age_at_capture: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Actual age of the subject (in years) at the time of recording. "
            "Required by the facettd module as a direct input. "
            "May be omitted when AnchorInput.birth_date is provided — "
            "it will be computed automatically."
        ),
    )


# ---------------------------------------------------------------------------
# Preprocessing output (persisted to disk)
# ---------------------------------------------------------------------------

class PreprocessedSample(BaseModel):
    """
    Result of preprocessing one VideoSample.

    All paths inside point to files that have already been written to
    `preprocess_dir` on disk — callers can inspect them directly.
    """

    subject_name: str
    captured_at: datetime
    age_at_capture: int

    face_image_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Deduplicated, frontal-face-filtered face images extracted from the video. "
            "At most `max_face_images` items (configurable in preprocess_pipeline)."
        ),
    )
    audio_segment_paths: list[str] = Field(
        default_factory=list,
        description="WAV audio clips attributed to the target speaker by VoiceTrack.",
    )
    preprocess_dir: str = Field(
        ...,
        description=(
            "Root directory where all intermediate files for this sample are stored. "
            "Layout: <uploads_root>/<subject_name>/<YYYYMMDD_HHMMSS>/"
        ),
    )


# ---------------------------------------------------------------------------
# Per-module result fragments (used inside SampleAnalysisResult)
# ---------------------------------------------------------------------------

class FaceAgeResult(BaseModel):
    biological_age_median: Optional[float] = None
    biological_age_values: list[float] = Field(default_factory=list)
    age_delta: Optional[float] = None  # biological_age_median - age_at_capture
    status: str = "ok"
    error: Optional[str] = None


class FacettdResult(BaseModel):
    remaining_life_median: Optional[float] = None  # years
    remaining_life_values: list[float] = Field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None


class DiseaseRiskResult(BaseModel):
    """Generic risk result for Parkinsons / lung_cancer / skin_disease."""

    module_id: str
    risk_label: Optional[str] = None        # e.g. "high", "low", or class name
    risk_probability: Optional[float] = None  # mean probability across clips/images
    vote_counts: dict[str, int] = Field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Single-sample aggregated analysis result
# ---------------------------------------------------------------------------

class SampleAnalysisResult(BaseModel):
    """Aggregated outputs from all health modules for one video sample."""

    subject_name: str
    captured_at: datetime
    age_at_capture: int

    face_age: Optional[FaceAgeResult] = None
    facettd: Optional[FacettdResult] = None
    parkinsons: Optional[DiseaseRiskResult] = None
    lung_cancer: Optional[DiseaseRiskResult] = None
    skin_disease: Optional[DiseaseRiskResult] = None

    extra_modules: dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all for any additional modules not explicitly typed above.",
    )

    llm_prompt: Optional[str] = Field(
        None, description="Rendered Jinja2 prompt sent to the LLM for this sample."
    )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class TimelinePoint(BaseModel):
    """One data point on the remaining-lifetime timeline."""

    captured_at: datetime
    age_at_capture: int
    remaining_life_estimate: float = Field(..., description="Estimated remaining years.")
    uncertainty_low: float = Field(
        ..., description="Lower bound of the uncertainty interval (years)."
    )
    uncertainty_high: float = Field(
        ..., description="Upper bound of the uncertainty interval (years)."
    )
    key_factors: list[str] = Field(
        default_factory=list,
        description="Main factors cited by the LLM (e.g. 'elevated biological age').",
    )
    summary: str = Field(
        "", description="One-paragraph human-readable assessment from the LLM."
    )
    raw_analysis: Optional[SampleAnalysisResult] = Field(
        None, description="Full module-level breakdown for this point."
    )


class LifetimeTimeline(BaseModel):
    """Complete remaining-lifetime timeline for one subject."""

    subject_name: str
    points: list[TimelinePoint] = Field(
        default_factory=list,
        description="Timeline points sorted ascending by captured_at.",
    )

    def sorted_points(self) -> list[TimelinePoint]:
        return sorted(self.points, key=lambda p: p.captured_at)
