"""
Timeline builder — turns a list of (VideoSample, SampleAnalysisResult) pairs
into a LifetimeTimeline by rendering prompts and (optionally) calling an LLM.

LLM integration:
  - If env var LLM_API_KEY is set, calls the OpenAI-compatible chat API
    (endpoint configurable via LLM_API_BASE, defaults to api.openai.com).
  - If LLM_API_KEY is NOT set, the TimelinePoint is built from module outputs
    directly (facettd median as estimate, wide uncertainty band) and the
    rendered prompt is stored in `summary` so users can paste it manually.

Same-day merging:
  When multiple video samples share the same calendar date, their
  SampleAnalysisResult values are averaged and a single TimelinePoint
  is produced for that date.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.orchestrator import load_prompt_template
from core.workflow_types import (
    AnchorInput,
    LifetimeTimeline,
    SampleAnalysisResult,
    TimelinePoint,
    VideoSample,
)

logger = logging.getLogger(__name__)

# ── Prompt rendering ──────────────────────────────────────────────────────────

def _build_workflow_prompt(analysis: SampleAnalysisResult) -> str:
    """
    Render a structured lifetime-estimation prompt for one SampleAnalysisResult.
    Uses the workflow-specific template from config/prompt_template.yaml.
    Falls back to a hard-coded template if the YAML template cannot be loaded.
    """
    try:
        template_str = load_prompt_template()
    except Exception:
        template_str = ""

    # If the stored template has our new workflow markers, use Jinja2 rendering
    if "{{ subject_name }}" in template_str or "{{ age_at_capture }}" in template_str:
        from jinja2 import Environment, BaseLoader, Undefined

        class _Silent(Undefined):
            def __str__(self): return ""
            def __getattr__(self, _): return self

        env = Environment(loader=BaseLoader(), undefined=_Silent)
        tmpl = env.from_string(template_str)
        return tmpl.render(_sample_to_template_ctx(analysis))

    # Otherwise fall back to a built-in workflow prompt
    return _builtin_prompt(analysis)


def _sample_to_template_ctx(a: SampleAnalysisResult) -> dict:
    """Convert SampleAnalysisResult to a flat dict for Jinja2 template rendering."""
    ctx: dict = {
        "subject_name": a.subject_name,
        "captured_at": a.captured_at.strftime("%Y-%m-%d"),
        "age_at_capture": a.age_at_capture,
    }
    if a.face_age and a.face_age.status == "ok":
        ctx["faceage_bio_age"] = a.face_age.biological_age_median
        ctx["faceage_delta"] = a.face_age.age_delta
    if a.facettd and a.facettd.status == "ok":
        ctx["facettd_remaining"] = a.facettd.remaining_life_median
        ctx["facettd_values"] = a.facettd.remaining_life_values
    if a.parkinsons and a.parkinsons.status == "ok":
        ctx["parkinsons_label"] = a.parkinsons.risk_label
        ctx["parkinsons_prob"] = a.parkinsons.risk_probability
    if a.lung_cancer and a.lung_cancer.status == "ok":
        ctx["lung_cancer_label"] = a.lung_cancer.risk_label
        ctx["lung_cancer_prob"] = a.lung_cancer.risk_probability
    if a.skin_disease and a.skin_disease.status == "ok":
        ctx["skin_disease_label"] = a.skin_disease.risk_label
        ctx["skin_disease_prob"] = a.skin_disease.risk_probability
    return ctx


def _builtin_prompt(a: SampleAnalysisResult) -> str:
    """Hard-coded fallback prompt template for workflow mode."""
    lines = [
        "你是一名经验丰富的老年医学专家。",
        "请根据以下多模态健康检测数据，估算该人物当前的预期剩余寿命，并输出一个 JSON 对象。",
        "",
        f"## 对象信息",
        f"- 姓名：{a.subject_name}",
        f"- 采样日期：{a.captured_at.strftime('%Y-%m-%d')}",
        f"- 拍摄时实际年龄：{a.age_at_capture} 岁",
        "",
    ]

    if a.face_age and a.face_age.status == "ok":
        lines += [
            "## 面部生物年龄（FaceAge）",
            f"- 生物年龄（中位数）：{a.face_age.biological_age_median:.1f} 岁",
            f"- 与实际年龄差：{a.face_age.age_delta:+.1f} 岁（正值=看起来偏老）",
            "",
        ]

    if a.facettd and a.facettd.status == "ok":
        lines += [
            "## 面部死亡时间预测（facettd）",
            f"- 剩余寿命估计（中位数）：{a.facettd.remaining_life_median:.1f} 年",
            "",
        ]

    if a.parkinsons and a.parkinsons.status == "ok":
        label = a.parkinsons.risk_label or "未知"
        prob = f"{a.parkinsons.risk_probability * 100:.1f}%" if a.parkinsons.risk_probability is not None else "N/A"
        lines += [
            "## 帕金森风险（Parkinsons）",
            f"- 风险标签：{label}",
            f"- 平均概率：{prob}",
            "",
        ]

    if a.lung_cancer and a.lung_cancer.status == "ok":
        label = a.lung_cancer.risk_label or "未知"
        prob = f"{a.lung_cancer.risk_probability * 100:.1f}%" if a.lung_cancer.risk_probability is not None else "N/A"
        lines += [
            "## 肺癌风险（LungCancer）",
            f"- 风险标签：{label}",
            f"- 平均概率：{prob}",
            "",
        ]

    if a.skin_disease and a.skin_disease.status == "ok":
        lines += [
            "## 皮肤病（SkinDisease，可选）",
            f"- 风险标签：{a.skin_disease.risk_label or '未知'}",
            "",
        ]

    lines += [
        "## 输出要求",
        "请仅输出以下 JSON 对象，不要包含任何 Markdown 代码块标记：",
        '{',
        '  "remaining_life_years": <估计剩余寿命（年，浮点数）>,',
        '  "uncertainty_low": <95%置信区间下界（年）>,',
        '  "uncertainty_high": <95%置信区间上界（年）>,',
        '  "key_factors": ["主要影响因素1", "主要影响因素2"],',
        '  "summary": "一段简短的综合评估说明（2-3句话）"',
        '}',
        "",
        "注意：",
        "- 若某项检测数据缺失，在评估中忽略该项，不要虚构数据。",
        "- 置信区间宽度应反映数据质量和不确定性。",
        "- 剩余寿命估计不代表任何临床结论，仅供研究参考。",
    ]

    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> Optional[dict]:
    """
    Call an OpenAI-compatible chat completion API.

    Required env vars:
        LLM_API_KEY   — API key (any value enables LLM mode)

    Optional env vars:
        LLM_API_BASE  — Base URL (default: https://api.openai.com/v1)
        LLM_MODEL     — Model name (default: gpt-4o)

    Returns parsed JSON dict from the LLM response, or None on failure.
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        return None

    api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


# ── TimelinePoint construction ────────────────────────────────────────────────

def _analysis_to_timeline_point(
    analysis: SampleAnalysisResult,
    llm_result: Optional[dict],
) -> TimelinePoint:
    """
    Build one TimelinePoint from module outputs + optional LLM result.

    If LLM result is available, its values take priority.
    If not, falls back to facettd median with a ±50% uncertainty band.
    """
    if llm_result and "remaining_life_years" in llm_result:
        remaining = float(llm_result["remaining_life_years"])
        unc_low = float(llm_result.get("uncertainty_low", max(0, remaining * 0.5)))
        unc_high = float(llm_result.get("uncertainty_high", remaining * 1.5))
        factors = llm_result.get("key_factors", [])
        summary = llm_result.get("summary", "")
    elif analysis.facettd and analysis.facettd.status == "ok" and analysis.facettd.remaining_life_median is not None:
        remaining = analysis.facettd.remaining_life_median
        unc_low = max(0.0, remaining * 0.5)
        unc_high = remaining * 1.5
        factors = ["面部死亡时间预测（无LLM）"]
        summary = f"基于 facettd 模型直接估计，剩余寿命约 {remaining:.1f} 年（无 LLM 整合）。"
    else:
        # Nothing to work with — mark as uncertain
        remaining = float("nan")
        unc_low = 0.0
        unc_high = 100.0
        factors = []
        summary = "数据不足，无法给出有效估计。"

    return TimelinePoint(
        captured_at=analysis.captured_at,
        age_at_capture=analysis.age_at_capture,
        remaining_life_estimate=remaining if remaining == remaining else -1.0,
        uncertainty_low=unc_low,
        uncertainty_high=unc_high,
        key_factors=factors,
        summary=summary,
        raw_analysis=analysis,
    )


# ── Same-day merging ──────────────────────────────────────────────────────────

def _merge_same_day_points(points: list[TimelinePoint]) -> list[TimelinePoint]:
    """
    Merge multiple TimelinePoints that share the same calendar date
    by averaging numeric estimates. Keeps the most complete raw_analysis.
    """
    by_date: defaultdict[date, list[TimelinePoint]] = defaultdict(list)
    for p in points:
        by_date[p.captured_at.date()].append(p)

    merged: list[TimelinePoint] = []
    for day, day_points in sorted(by_date.items()):
        if len(day_points) == 1:
            merged.append(day_points[0])
            continue

        valid = [p for p in day_points if p.remaining_life_estimate >= 0]
        if not valid:
            merged.append(day_points[0])
            continue

        avg_remaining = statistics.mean(p.remaining_life_estimate for p in valid)
        avg_low = statistics.mean(p.uncertainty_low for p in valid)
        avg_high = statistics.mean(p.uncertainty_high for p in valid)

        # Collect all unique factors
        all_factors: list[str] = []
        seen: set[str] = set()
        for p in day_points:
            for f in p.key_factors:
                if f not in seen:
                    all_factors.append(f)
                    seen.add(f)

        summary = f"（{len(day_points)} 个样本合并）" + (valid[0].summary if valid else "")

        merged.append(TimelinePoint(
            captured_at=day_points[0].captured_at,
            age_at_capture=day_points[0].age_at_capture,
            remaining_life_estimate=avg_remaining,
            uncertainty_low=avg_low,
            uncertainty_high=avg_high,
            key_factors=all_factors,
            summary=summary,
            raw_analysis=day_points[0].raw_analysis,
        ))

    return sorted(merged, key=lambda p: p.captured_at)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_timeline(
    subject_name: str,
    analyses: list[SampleAnalysisResult],
    progress_cb: Optional[Callable] = None,
) -> LifetimeTimeline:
    """
    Build a LifetimeTimeline from one or more SampleAnalysisResult objects.

    For each analysis:
      1. Render a structured LLM prompt.
      2. Call LLM if LLM_API_KEY is set; otherwise use module outputs directly.
      3. Construct a TimelinePoint.
    Then merge same-day points and sort by date.

    Args:
        subject_name: Human-readable name / ID for the subject.
        analyses:     List of per-video analysis results (any order).

    Returns:
        LifetimeTimeline sorted ascending by captured_at.
    """
    def _cb(event_type: str, **data):
        if progress_cb:
            try:
                progress_cb(event_type, **data)
            except Exception:
                pass

    has_llm = bool(os.environ.get("LLM_API_KEY", "").strip())
    points: list[TimelinePoint] = []

    for analysis in analyses:
        day = analysis.captured_at.strftime("%Y-%m-%d")
        logger.info("[%s] 构建时间轴点 @ %s …", subject_name, day)

        prompt = _build_workflow_prompt(analysis)
        analysis.llm_prompt = prompt

        if has_llm:
            _cb("phase", phase="llm",
                desc=f"调用 LLM 综合分析 {day} 的数据…")
        else:
            _cb("phase", phase="llm",
                desc=f"LLM 未配置，直接使用 facettd 估计值（{day}）")

        llm_result = _call_llm(prompt)

        if llm_result:
            logger.info("[%s] LLM 响应成功", subject_name)
            _cb("step", step="llm_done", status="ok",
                desc=f"✓ LLM 综合分析完成（{day}）")
        else:
            logger.info("[%s] LLM 未配置或调用失败，使用模块直接输出", subject_name)
            _cb("step", step="llm_done", status="fallback",
                desc=f"✓ 使用 facettd 直接估计（{day}，无 LLM）")

        point = _analysis_to_timeline_point(analysis, llm_result)
        points.append(point)

    merged_points = _merge_same_day_points(points)

    return LifetimeTimeline(subject_name=subject_name, points=merged_points)
