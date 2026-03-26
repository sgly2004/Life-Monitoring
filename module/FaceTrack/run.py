#!/usr/bin/env python3
"""
FaceTrack — Video Face Tracking and Clipping

Input  (stdin JSON):
  {
    "video_path": "...",
    "reference_face_path": "...",
    "threshold": 0.70
  }

Output (stdout JSON): UnifiedResult schema
"""
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_tracker import track_faces_in_video

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_TEMPLATE = {
    "module_id": "facetrack",
    "module_name": "视频人脸切分 (FaceTrack)",
    "status": "error",
    "inputs": {},
    "outputs": {},
    "summary": "",
    "error": None,
}


def main():
    data = json.load(sys.stdin)

    video_path = data.get("video_path", "")
    ref_face_path = data.get("reference_face_path", "")
    threshold = float(data.get("threshold", 0.70))
    frame_interval = int(data.get("frame_interval", 5))
    min_face_px = int(data.get("min_face_px", 60))

    result = dict(RESULT_TEMPLATE)
    result["inputs"] = {
        "video": os.path.basename(video_path),
        "reference_face": os.path.basename(ref_face_path),
        "threshold": threshold,
    }

    if not video_path:
        result["error"] = "video_path is required"
        result["summary"] = "FaceTrack 运行失败：缺少视频路径"
        print(json.dumps(result, ensure_ascii=False))
        return

    if not ref_face_path:
        result["error"] = "reference_face_path is required"
        result["summary"] = "FaceTrack 运行失败：缺少参考人脸路径"
        print(json.dumps(result, ensure_ascii=False))
        return

    if not os.path.exists(video_path):
        result["error"] = f"Video file not found: {video_path}"
        result["summary"] = f"FaceTrack 运行失败：视频文件不存在"
        print(json.dumps(result, ensure_ascii=False))
        return

    if not os.path.exists(ref_face_path):
        result["error"] = f"Reference face image not found: {ref_face_path}"
        result["summary"] = f"FaceTrack 运行失败：参考人脸图片不存在"
        print(json.dumps(result, ensure_ascii=False))
        return

    try:
        clips, total_duration = track_faces_in_video(
            video_path=video_path,
            reference_face_path=ref_face_path,
            threshold=threshold,
            frame_interval=frame_interval,
            min_face_px=min_face_px,
        )

        total_clips = len(clips)
        result["status"] = "success"
        result["outputs"] = {
            "clips": clips,
            "total_clips": total_clips,
            "total_duration": round(total_duration, 1),
            "target_identity": "target_person",
            "video_path": video_path,
        }
        result["summary"] = (
            f"在视频中检测到 {total_clips} 个目标人物片段，"
            f"总时长 {total_duration:.1f} 秒"
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"FaceTrack 运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
