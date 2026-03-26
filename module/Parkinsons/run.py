#!/usr/bin/env python3
"""
Parkinsons — Parkinson's Disease Acoustic Detector

Input  (stdin JSON) : {"audio_path": "..."}
Output (stdout JSON): UnifiedResult schema
"""
import sys
import json
import os
import warnings

warnings.filterwarnings("ignore")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(MODULE_DIR, "parkinsons.csv")

RESULT = {
    "module_id": "parkinsons",
    "module_name": "帕金森声学检测 (Parkinsons)",
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
        from detect import detect

        detection = detect(audio_path, CSV_PATH)

        label = detection["label"]
        p_parkinson = detection["prob_parkinson"]
        p_healthy = detection["prob_healthy"]
        diagnosis = detection["diagnosis"]

        if p_parkinson > 0.75:
            risk = "高风险"
        elif p_parkinson > 0.5:
            risk = "中等风险"
        else:
            risk = "低风险"

        result["status"] = "success"
        result["outputs"] = {
            "diagnosis": diagnosis,
            "prob_parkinson": p_parkinson,
            "prob_healthy": p_healthy,
            "risk_level": risk,
            "label": label,
        }
        result["summary"] = (
            f"帕金森检测结果：{diagnosis}（风险等级：{risk}），"
            f"帕金森概率 {p_parkinson * 100:.1f}%，正常概率 {p_healthy * 100:.1f}%。"
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"Parkinsons 模块运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
