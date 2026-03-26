#!/usr/bin/env python3
"""
Early Stage Lung Cancer Detection from Speech Sounds

Input  (stdin JSON) : {"audio_path": "..."}
Output (stdout JSON): UnifiedResult schema

Prerequisite: run main_codes/run_pipeline.py once to train and save models.
"""
import sys
import json
import os
import warnings

warnings.filterwarnings("ignore")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT = {
    "module_id": "lung_cancer",
    "module_name": "早期肺癌语音检测 (Lung Cancer)",
    "status": "error",
    "inputs": {},
    "outputs": {},
    "summary": "",
    "error": None,
}


def main():
    data = json.load(sys.stdin)
    audio_path = data.get("audio_path", "")

    result = dict(RESULT)
    result["inputs"] = {"audio": os.path.basename(audio_path)}

    try:
        sys.path.insert(0, MODULE_DIR)
        from infer import infer

        detection = infer(audio_path)

        result["status"] = "success"
        result["outputs"] = {
            "final_verdict": detection["final_verdict"],
            "is_positive": detection["is_positive"],
            "votes_positive": detection["votes_positive"],
            "total_models": detection["total_models"],
            "segments_analyzed": detection["segments_analyzed"],
            "model_results": detection["model_results"],
        }

        votes = detection["votes_positive"]
        total = detection["total_models"]
        verdict = detection["final_verdict"]
        segs = detection["segments_analyzed"]

        result["summary"] = (
            f"肺癌语音检测结果：{verdict}（{votes}/{total} 个模型阳性，"
            f"分析音频片段 {segs} 段）。"
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"Lung Cancer 模块运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
