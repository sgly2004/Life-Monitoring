#!/usr/bin/env python3
"""
VoiceTrack — Audio Voice Identity Tracking and Clipping

Input  (stdin JSON):
  {
    "audio_path": "...",
    "reference_audio_path": "...",
    "threshold": 0.75
  }

Output (stdout JSON): UnifiedResult schema
"""
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_tracker import track_voice_in_audio

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_TEMPLATE = {
    "module_id": "voicetrack",
    "module_name": "音频声纹切分 (VoiceTrack)",
    "status": "error",
    "inputs": {},
    "outputs": {},
    "summary": "",
    "error": None,
}


def main():
    data = json.load(sys.stdin)

    audio_path = data.get("audio_path", "")
    ref_audio_path = data.get("reference_audio_path", "")
    threshold = float(data.get("threshold", 0.75))

    result = dict(RESULT_TEMPLATE)
    result["inputs"] = {
        "audio": os.path.basename(audio_path),
        "reference_audio": os.path.basename(ref_audio_path),
        "threshold": threshold,
    }

    if not audio_path:
        result["error"] = "audio_path is required"
        result["summary"] = "VoiceTrack 运行失败：缺少音频路径"
        print(json.dumps(result, ensure_ascii=False))
        return

    if not ref_audio_path:
        result["error"] = "reference_audio_path is required"
        result["summary"] = "VoiceTrack 运行失败：缺少参考语音路径"
        print(json.dumps(result, ensure_ascii=False))
        return

    if not os.path.exists(audio_path):
        result["error"] = f"Audio file not found: {audio_path}"
        result["summary"] = f"VoiceTrack 运行失败：音频文件不存在"
        print(json.dumps(result, ensure_ascii=False))
        return

    if not os.path.exists(ref_audio_path):
        result["error"] = f"Reference audio not found: {ref_audio_path}"
        result["summary"] = f"VoiceTrack 运行失败：参考语音文件不存在"
        print(json.dumps(result, ensure_ascii=False))
        return

    try:
        clips, total_duration = track_voice_in_audio(
            audio_path=audio_path,
            reference_audio_path=ref_audio_path,
            threshold=threshold,
        )

        total_clips = len(clips)
        result["status"] = "success"
        result["outputs"] = {
            "clips": clips,
            "total_clips": total_clips,
            "total_duration": round(total_duration, 2),
            "target_speaker": "target_speaker",
        }
        result["summary"] = (
            f"在音频中检测到 {total_clips} 个目标说话人片段，"
            f"总时长 {total_duration:.1f} 秒"
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"VoiceTrack 运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
