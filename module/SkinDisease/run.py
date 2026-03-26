#!/usr/bin/env python3
"""
SkinDisease — Skin Lesion Diagnosis (run.py)
Unified entry point for the Life-Monitoring orchestrator.

Input  (stdin JSON): {"image_path": "..."}
Output (stdout JSON): UnifiedResult schema
"""
import sys
import json
import os
import warnings

warnings.filterwarnings("ignore")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(MODULE_DIR, "model")

RESULT = {
    "module_id": "skin_disease",
    "module_name": "皮肤病检测 (SkinDisease)",
    "status": "error",
    "inputs": {},
    "outputs": {},
    "summary": "",
    "error": None,
}

# Severity levels used in the LLM summary
SEVERITY_CN = {
    "低": "低风险",
    "中": "中等风险",
    "高": "高风险",
}

SEVERITY_MAP = {
    "酒糟鼻（玫瑰痤疮）": "中",
    "粟丘疹（白头）": "低",
    "角化病（脂溢性角化病）": "低",
    "湿疹": "中",
    "皮肤癌（鳞状/基底细胞癌）": "高",
    "痤疮": "低",
}


def main():
    data = json.load(sys.stdin)
    image_path = data.get("image_path", "")

    result = dict(RESULT)
    result["inputs"] = {"image": os.path.basename(image_path)}

    try:
        sys.path.insert(0, MODULE_DIR)
        from predict import predict

        if not os.path.exists(os.path.join(MODEL_DIR, "skin_model.keras")):
            raise FileNotFoundError(
                f"模型文件不存在：{MODEL_DIR}/skin_model.keras\n"
                "请先运行 README.md 中的模型下载说明。"
            )

        detection = predict(image_path)

        top_class = detection["top_class"]
        confidence = detection["confidence"]
        severity = SEVERITY_MAP[detection["diagnosis"]]
        all_probs = detection["all_probabilities"]

        # Build readable probability summary string
        prob_lines = "、".join(
            f"{p['label_cn']} {p['probability'] * 100:.1f}%"
            for p in all_probs[:3]
        )

        result["status"] = "success"
        result["outputs"] = {
            "diagnosis": detection["diagnosis"],
            "diagnosis_en": detection["diagnosis_en"],
            "confidence": confidence,
            "severity": SEVERITY_CN[severity],
            "urgency": detection["urgency"],
            "top_class": top_class,
            "all_probabilities": all_probs,
        }
        result["summary"] = (
            f"皮肤病检测结果：{detection['diagnosis']}，"
            f"置信度 {confidence * 100:.1f}%，"
            f"严重程度：{SEVERITY_CN[severity]}。"
            f"前三候选：{prob_lines}。"
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"SkinDisease 模块运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
