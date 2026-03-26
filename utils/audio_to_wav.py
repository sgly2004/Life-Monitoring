"""
audio_to_wav.py
---------------
将常见音频格式（mp3 / m4a / aac / flac / ogg / opus / wma / aiff 等）
批量或单文件转换为 WAV，供声学特征提取使用。

依赖：ffmpeg（系统工具，需提前安装）
    macOS:   brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg
    Windows: https://ffmpeg.org/download.html

用法（命令行）：
    # 转换单个文件
    python audio_to_wav.py data/sound/20260312_151645.m4a

    # 转换并指定输出路径
    python audio_to_wav.py input.mp3 -o output/patient_001.wav

    # 批量转换整个目录
    python audio_to_wav.py data/sound/ --batch

    # 批量转换并指定采样率（帕金森特征提取推荐 44100）
    python audio_to_wav.py data/sound/ --batch --sr 44100

    # 静默模式（不打印进度）
    python audio_to_wav.py data/sound/ --batch --quiet

用法（Python 导入）：
    from utils.audio_to_wav import to_wav, batch_to_wav

    out = to_wav("data/sound/20260312_151645.m4a")
    print(out)  # data/sound/20260312_151645.wav

    results = batch_to_wav("data/sound/", sample_rate=44100)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ffmpeg 支持的常见输入格式
SUPPORTED_FORMATS = {
    ".mp3", ".m4a", ".aac", ".flac", ".ogg",
    ".opus", ".wma", ".aiff", ".aif", ".webm",
    ".mp4", ".mov", ".mkv",  # 视频格式（提取音轨）
}


def _check_ffmpeg() -> str:
    """找到 ffmpeg 可执行路径，找不到则报错。"""
    path = shutil.which("ffmpeg")
    if path is None:
        sys.exit(
            "[错误] 未找到 ffmpeg。\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    return path


def to_wav(
    input_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 44100,
    channels: int = 1,
    overwrite: bool = True,
    quiet: bool = False,
) -> Path:
    """
    将单个音频文件转换为 WAV。

    参数
    ----
    input_path  : 输入文件路径（支持 mp3 / m4a / aac / flac / ogg 等）
    output_path : 输出 WAV 路径；默认与输入同目录、同文件名、后缀改为 .wav
    sample_rate : 目标采样率（Hz），帕金森特征提取建议 44100
    channels    : 声道数，1=单声道（Praat 分析用），2=立体声
    overwrite   : True 则覆盖已存在的输出文件
    quiet       : True 则不打印进度信息

    返回
    ----
    Path  输出文件的绝对路径
    """
    ffmpeg = _check_ffmpeg()
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        if not quiet:
            print(f"[跳过] 已存在：{output_path}")
        return output_path

    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",   # 覆盖 / 不覆盖
        "-i", str(input_path),         # 输入
        "-ar", str(sample_rate),       # 采样率
        "-ac", str(channels),          # 声道数
        "-vn",                         # 不处理视频流
        "-acodec", "pcm_s16le",        # 16-bit PCM，标准 WAV
        str(output_path),
    ]

    run_kwargs = dict(check=True, capture_output=True, text=True)
    try:
        subprocess.run(cmd, **run_kwargs)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg 转换失败：{input_path.name}\n{e.stderr}"
        ) from e

    if not quiet:
        size_kb = output_path.stat().st_size / 1024
        print(f"[完成] {input_path.name} → {output_path.name}  ({size_kb:.1f} KB)")

    return output_path


def batch_to_wav(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    sample_rate: int = 44100,
    channels: int = 1,
    overwrite: bool = True,
    quiet: bool = False,
) -> list[Path]:
    """
    批量转换目录下所有受支持的音频文件为 WAV。

    参数
    ----
    input_dir  : 包含音频文件的目录
    output_dir : 输出目录；默认与 input_dir 相同（原地转换）
    sample_rate: 目标采样率（Hz）
    channels   : 声道数
    overwrite  : 是否覆盖已存在的 WAV 文件
    quiet      : 静默模式

    返回
    ----
    list[Path]  成功转换的输出文件路径列表
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"不是有效目录：{input_dir}")

    candidates = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_FORMATS and p.suffix.lower() != ".wav"
    ]

    if not candidates:
        if not quiet:
            print(f"[提示] {input_dir} 中没有找到可转换的音频文件。")
        return []

    if not quiet:
        print(f"[批量转换] 找到 {len(candidates)} 个文件，采样率={sample_rate} Hz，声道={channels}")

    results = []
    errors = []
    for src in sorted(candidates):
        dst = (Path(output_dir) / src.with_suffix(".wav").name) if output_dir else None
        try:
            out = to_wav(src, dst, sample_rate, channels, overwrite, quiet)
            results.append(out)
        except Exception as exc:
            errors.append((src, exc))
            print(f"[失败] {src.name}: {exc}")

    if not quiet:
        print(f"\n转换完成：{len(results)} 成功，{len(errors)} 失败。")

    return results


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将音频文件（mp3/m4a/aac/flac 等）转换为 WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="输入文件或目录路径")
    p.add_argument("-o", "--output", default=None,
                   help="输出 WAV 路径（单文件模式）或输出目录（批量模式）")
    p.add_argument("--batch", action="store_true",
                   help="批量模式：转换整个目录")
    p.add_argument("--sr", "--sample-rate", type=int, default=44100,
                   dest="sample_rate",
                   help="目标采样率，默认 44100 Hz")
    p.add_argument("--channels", type=int, default=1,
                   help="声道数，默认 1（单声道）")
    p.add_argument("--no-overwrite", action="store_false", dest="overwrite",
                   help="不覆盖已存在的 WAV 文件")
    p.add_argument("--quiet", action="store_true",
                   help="不输出进度信息")
    return p


def main():
    args = _build_parser().parse_args()

    if args.batch or Path(args.input).is_dir():
        batch_to_wav(
            args.input,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
    else:
        to_wav(
            args.input,
            output_path=args.output,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
