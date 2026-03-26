"""
video_to_audio.py
-----------------
从视频文件中截取指定时间段并提取为 WAV 音频，保存到 data/sound/。

依赖：ffmpeg（系统工具）
    macOS:   brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg
    Windows: https://ffmpeg.org/download.html

用法（命令行）：
    # 提取 0:30 ~ 1:15 之间的音频
    python utils/video_to_audio.py interview.mp4 --start 00:00:30 --end 00:01:15

    # 用秒数指定时间
    python utils/video_to_audio.py interview.mp4 --start 30 --end 75

    # 指定输出路径
    python utils/video_to_audio.py interview.mp4 --start 30 --end 75 -o data/sound/segment.wav

    # 提取全段音频（不裁剪）
    python utils/video_to_audio.py interview.mp4

    # 指定采样率（帕金森/肺癌模块推荐 44100）
    python utils/video_to_audio.py interview.mp4 --start 10 --end 40 --sr 44100

用法（Python 导入）：
    from utils.video_to_audio import extract_audio

    out = extract_audio("data/video/patient.mp4", start="00:01:00", end="00:01:30")
    print(out)  # data/sound/patient_60.0-90.0.wav

    # 只指定 start（提取到结尾）
    out = extract_audio("data/video/patient.mp4", start=30)

    # 全段提取
    out = extract_audio("data/video/patient.mp4")
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# 默认输出目录（相对于本文件所在项目根目录）
_HERE = Path(__file__).resolve().parent          # utils/
ROOT_DIR = _HERE.parent                           # Life-Monitoring/
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "sound"

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".wmv",
    ".flv", ".webm", ".m4v", ".3gp", ".ts",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        sys.exit(
            "[错误] 未找到 ffmpeg。\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    return path


def _parse_time(t: str | float | int | None) -> float | None:
    """
    将时间描述转换为秒数（float）。

    支持格式：
        90          → 90.0
        "90"        → 90.0
        "1:30"      → 90.0
        "00:01:30"  → 90.0
        "1:30.5"    → 90.5
    返回 None 表示未指定时间。
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    t = t.strip()
    # HH:MM:SS.ms or MM:SS.ms or SS.ms
    m = re.fullmatch(
        r"(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)",
        t,
    )
    if not m:
        raise ValueError(
            f"无法解析时间 '{t}'。"
            "支持格式：秒数（90）、MM:SS（1:30）、HH:MM:SS（00:01:30）"
        )
    parts = [g for g in m.groups() if g is not None]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def _seconds_to_hhmmss(s: float) -> str:
    """90.5 → '00:01:30.500'"""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _stem_with_time(video_stem: str, start_s: float | None, end_s: float | None) -> str:
    """Build an output filename stem that encodes the time range."""
    if start_s is None and end_s is None:
        return video_stem
    start_tag = f"{start_s:.1f}" if start_s is not None else "0"
    end_tag   = f"{end_s:.1f}"   if end_s   is not None else "end"
    return f"{video_stem}_{start_tag}-{end_tag}"


# ── main API ──────────────────────────────────────────────────────────────────

def extract_audio(
    video_path: str | Path,
    start: str | float | int | None = None,
    end:   str | float | int | None = None,
    output_path: str | Path | None = None,
    sample_rate: int = 44100,
    channels: int = 1,
    overwrite: bool = True,
    quiet: bool = False,
) -> Path:
    """
    从视频中截取指定时间段并提取为 WAV 音频。

    参数
    ----
    video_path  : 输入视频文件路径（支持 mp4 / mov / mkv / avi 等）
    start       : 起始时间。支持秒数（float/int）或字符串（"MM:SS" / "HH:MM:SS"）。
                  None 表示从头开始。
    end         : 终止时间。同上。None 表示到结尾。
    output_path : 输出 WAV 文件路径。
                  默认保存到 data/sound/<视频名>_<start>-<end>.wav
    sample_rate : 采样率（Hz），默认 44100（帕金森/肺癌特征提取推荐值）
    channels    : 声道数，1=单声道，2=立体声。默认 1。
    overwrite   : 是否覆盖已存在的输出文件。默认 True。
    quiet       : 不打印进度信息。

    返回
    ----
    Path  输出 WAV 文件的绝对路径
    """
    ffmpeg = _check_ffmpeg()
    video_path = Path(video_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    start_s = _parse_time(start)
    end_s   = _parse_time(end)

    if start_s is not None and end_s is not None and end_s <= start_s:
        raise ValueError(
            f"终止时间 ({end_s}s) 必须大于起始时间 ({start_s}s)"
        )

    # Build output path
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = _stem_with_time(video_path.stem, start_s, end_s)
        output_path = DEFAULT_OUTPUT_DIR / f"{stem}.wav"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        if not quiet:
            print(f"[跳过] 已存在：{output_path}")
        return output_path

    # Build ffmpeg command
    cmd = [ffmpeg, "-y" if overwrite else "-n"]

    # Seek before input for accuracy (fast seek when start is specified)
    if start_s is not None:
        cmd += ["-ss", _seconds_to_hhmmss(start_s)]

    cmd += ["-i", str(video_path)]

    # Duration or end timestamp (relative to start)
    if end_s is not None:
        duration = end_s - (start_s or 0.0)
        cmd += ["-t", f"{duration:.3f}"]

    cmd += [
        "-vn",                    # discard video stream
        "-ar", str(sample_rate),  # audio sample rate
        "-ac", str(channels),     # channels
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        str(output_path),
    ]

    if not quiet:
        time_desc = ""
        if start_s is not None or end_s is not None:
            s_str = _seconds_to_hhmmss(start_s) if start_s is not None else "开头"
            e_str = _seconds_to_hhmmss(end_s)   if end_s   is not None else "结尾"
            time_desc = f"  [{s_str} → {e_str}]"
        print(f"[提取音频] {video_path.name}{time_desc} → {output_path.name}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg 提取失败：{video_path.name}\n{e.stderr[-1000:]}"
        ) from e

    if not quiet:
        size_kb = output_path.stat().st_size / 1024
        print(f"[完成] {output_path}  ({size_kb:.1f} KB)")

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从视频中截取指定时间段并提取为 WAV 音频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", help="输入视频文件路径")
    p.add_argument(
        "--start", "-s", default=None,
        help="起始时间，如 30、1:30、00:01:30（默认从头）",
    )
    p.add_argument(
        "--end", "-e", default=None,
        help="终止时间，如 75、1:15、00:01:15（默认到结尾）",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help=f"输出 WAV 路径（默认：data/sound/<视频名>_<start>-<end>.wav）",
    )
    p.add_argument(
        "--sr", "--sample-rate", type=int, default=44100, dest="sample_rate",
        help="采样率（Hz），默认 44100",
    )
    p.add_argument(
        "--channels", type=int, default=1,
        help="声道数（1=单声道，2=立体声），默认 1",
    )
    p.add_argument(
        "--no-overwrite", action="store_false", dest="overwrite",
        help="不覆盖已存在的输出文件",
    )
    p.add_argument("--quiet", action="store_true", help="静默模式")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    extract_audio(
        video_path=args.video,
        start=args.start,
        end=args.end,
        output_path=args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
