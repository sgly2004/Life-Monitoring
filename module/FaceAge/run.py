#!/usr/bin/env python3
"""
FaceAge — Biological Age Estimator

Single mode  (stdin JSON): {"image_path": "..."}
Batch  mode  (stdin JSON): {"image_paths": ["...", "...", ...]}
Output (stdout JSON): UnifiedResult  |  list of UnifiedResult
"""
import sys
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, "models", "faceage_model.h5")


def _patch_func_load():
    import tf_keras.src.utils.generic_utils as _gu

    _orig = _gu.func_load

    def _patched(code, globs=None, locs=None, closure=None):
        try:
            return _orig(code, globs, locs, closure)
        except EOFError:
            return lambda inputs, scale=0.2: inputs[0] + inputs[1] * scale

    _gu.func_load = _patched


def _predict_one(image_path: str, detector, model) -> dict:
    """Run prediction on a single image. Returns a result dict."""
    import numpy as np
    import PIL.Image
    from skimage.io import imread

    result = {
        "module_id": "faceage",
        "module_name": "面部生物年龄预测 (FaceAge)",
        "status": "error",
        "inputs": {"image": os.path.basename(image_path)},
        "outputs": {},
        "summary": "",
        "error": None,
    }

    try:
        img_arr = imread(image_path)
        img_rgb = np.array(PIL.Image.fromarray(img_arr).convert("RGB"))

        detections = detector.detect_faces(img_rgb)
        if not detections:
            raise RuntimeError("图像中未检测到人脸")

        det = detections[0]
        x1, y1, w, h = det["box"]
        x1, y1 = abs(x1), abs(y1)
        face = img_rgb[y1: y1 + h, x1: x1 + w]
        face_pil = PIL.Image.fromarray(face.astype("uint8")).convert("RGB")
        face_resized = np.asarray(face_pil.resize((160, 160)))
        face_norm = (face_resized - face_resized.mean()) / (face_resized.std() + 1e-8)
        face_input = face_norm.reshape(1, 160, 160, 3)

        faceage = float(model.predict(face_input, verbose=0).squeeze())
        confidence = round(float(det.get("confidence", 0)), 3)

        result["status"] = "success"
        result["outputs"] = {
            "biological_age": round(faceage, 1),
            "detection_confidence": confidence,
        }
        result["summary"] = (
            f"面部生物年龄估计为 {faceage:.1f} 岁（置信度: {confidence}）。"
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"FaceAge 运行出错：{exc}"

    return result


def main():
    data = json.load(sys.stdin)

    # Normalise to a list of paths
    if "image_paths" in data:
        image_paths = data["image_paths"]
        batch_mode = True
    else:
        image_paths = [data.get("image_path", "")]
        batch_mode = False

    try:
        _patch_func_load()
        import tf_keras as keras
        import mtcnn

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"模型文件不存在：{MODEL_PATH}\n"
                "下载：https://github.com/AIM-Harvard/FaceAge/releases/download/v1/faceage_model.h5"
            )

        model    = keras.models.load_model(MODEL_PATH)
        detector = mtcnn.MTCNN()

        results = [_predict_one(p, detector, model) for p in image_paths]

    except Exception as exc:
        # Model load failure → mark all as error
        results = [
            {
                "module_id": "faceage",
                "module_name": "面部生物年龄预测 (FaceAge)",
                "status": "error",
                "inputs": {"image": os.path.basename(p)},
                "outputs": {},
                "summary": f"模型加载失败：{exc}",
                "error": str(exc),
            }
            for p in image_paths
        ]

    if batch_mode:
        print(json.dumps(results, ensure_ascii=False))
    else:
        print(json.dumps(results[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
