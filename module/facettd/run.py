#!/usr/bin/env python3
"""
facettd — Face Time-to-Death Predictor

Input  (stdin JSON) : {"image_path": "...", "age": 65}
Output (stdout JSON): UnifiedResult schema
"""
import sys
import json
import os
import warnings

warnings.filterwarnings("ignore")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT = {
    "module_id": "facettd",
    "module_name": "面部死亡时间预测 (FaceTTD)",
    "status": "error",
    "inputs": {},
    "outputs": {},
    "summary": "",
    "error": None,
}


def main():
    data = json.load(sys.stdin)
    image_path = data.get("image_path", "")
    age = data.get("age")

    result = dict(RESULT)
    result["inputs"] = {"image": os.path.basename(image_path), "age": age}

    try:
        import numpy as np
        import joblib
        from PIL import Image

        scaler_path = os.path.join(MODULE_DIR, "age_scaler.pkl")
        age_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        img = Image.open(image_path).convert("L").resize((64, 64))
        img_features = np.array(img).flatten()

        scaled_age = (
            age_scaler.transform(np.array([[age]]))[0, 0]
            if age_scaler is not None
            else age
        )
        base_features = np.concatenate([img_features, [scaled_age, 0]])

        model_files = ["best_xgb.pkl", "reg_xgb_ttd.pkl", "reg_rf_ttd.pkl"]
        prediction = None
        model_used = None
        for mf in model_files:
            model_path = os.path.join(MODULE_DIR, mf)
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            dim = getattr(model, "n_features_in_", len(base_features))
            if len(base_features) < dim:
                features = np.concatenate([base_features, np.zeros(dim - len(base_features))])
            elif len(base_features) > dim:
                features = base_features[:dim]
            else:
                features = base_features
            prediction = float(model.predict(features.reshape(1, -1))[0])
            model_used = mf
            break

        if prediction is None:
            raise FileNotFoundError(
                "未找到训练好的模型文件。请先在 module/facettd/ 目录下放置 .pkl 模型文件。"
            )

        life_expectancy = age + prediction
        result["status"] = "success"
        result["outputs"] = {
            "time_to_death_years": round(prediction, 1),
            "life_expectancy_age": round(life_expectancy, 1),
            "model_used": model_used,
        }
        result["summary"] = (
            f"基于面部特征（模型: {model_used}），预测剩余寿命约 {prediction:.1f} 年，"
            f"预期寿命约 {life_expectancy:.1f} 岁（当前年龄 {age} 岁）。"
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"facettd 模块运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
