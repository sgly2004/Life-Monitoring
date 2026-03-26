#!/usr/bin/env python3
"""
SkinDisease — Skin Lesion Diagnosis (predict.py)

Based on Tanishq77/skin-condition-classifier (Hugging Face):
  https://huggingface.co/Tanishq77/skin-condition-classifier

6-class classifier (EfficientNetV2B0 backbone, 95.6% test accuracy):
  0: rosacea   (酒糟鼻/玫瑰痤疮)
  1: mila      (粟丘疹/白头)
  2: keratosis (角化病/脂溢性角化病)
  3: eczema    (湿疹)
  4: carcinoma (皮肤癌/基底细胞癌)
  5: acne      (痤疮)

Input : image_path (str) — any PIL-readable image file
Output: dict with diagnosis, class probabilities, confidence
"""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from PIL import Image

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, "model", "skin_model.keras")

CLASS_LABELS_CN = [
    "酒糟鼻（玫瑰痤疮）",
    "粟丘疹（白头）",
    "角化病（脂溢性角化病）",
    "湿疹",
    "皮肤癌（鳞状/基底细胞癌）",
    "痤疮",
]

CLASS_LABELS_EN = [
    "rosacea",
    "mila",
    "keratosis",
    "eczema",
    "carcinoma",
    "acne",
]

# Severity / urgency mapping for risk assessment
SEVERITY = {
    0: "中",      # rosacea  — chronic, manageable
    1: "低",      # mila     — benign cyst
    2: "低",      # keratosis — usually benign
    3: "中",      # eczema   — chronic inflammatory
    4: "高",      # carcinoma — potentially malignant, requires attention
    5: "低",      # acne     — common, usually benign
}

URGENCY = {
    0: "建议皮肤科就诊，配合药物与生活方式调整",
    1: "一般无需特殊处理，必要时可美容科就诊",
    2: "建议皮肤科随诊观察",
    3: "建议皮肤科就诊，日常护理与药物治疗可缓解",
    4: "建议尽快皮肤科就诊，必要时活检排除恶性病变",
    5: "建议门诊皮肤科随诊",
}


def load_model(model_path: str = MODEL_PATH):
    """Load the saved EfficientNetV2B0 model."""
    import keras

    model = keras.models.load_model(model_path)
    return model


def _preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess a single image for EfficientNetV2B0.
    1. Load with PIL (auto-corrects orientation/exif)
    2. Resize to target_size
    3. Convert to numpy array (HWC, float32)
    4. Batch dimension
    5. EfficientNetV2 preprocess (scales pixels to [-1, 1])
    """
    import keras.applications.efficientnet_v2 as efficientnet_v2

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = efficientnet_v2.preprocess_input(arr)
    return arr


def predict(image_path: str, model=None) -> dict:
    """
    Run skin disease classification on a single image.

    Args:
        image_path : path to the input image
        model      : optional pre-loaded model (avoids reload cost)

    Returns:
        {
            "diagnosis"        : str,  top-1 disease name (Chinese)
            "diagnosis_en"     : str,  top-1 disease name (English)
            "top_class"        : int,  top-1 class index (0-5)
            "confidence"       : float, top-1 softmax probability
            "all_probabilities": list[dict], all 6 classes with probability %
            "severity"         : str,  severity level (低/中/高)
            "urgency"          : str,  recommended next step (Chinese)
        }
    """
    import keras.applications.efficientnet_v2 as efficientnet_v2

    if model is None:
        model = load_model(MODEL_PATH)

    x = _preprocess_image(image_path)
    raw = model.predict(x, verbose=0)
    probs = np.squeeze(raw).astype(float)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    all_classes = []
    for i, (label_cn, label_en, prob) in enumerate(
        zip(CLASS_LABELS_CN, CLASS_LABELS_EN, probs.tolist())
    ):
        all_classes.append({
            "class_idx": i,
            "label_cn": label_cn,
            "label_en": label_en,
            "probability": round(float(prob), 4),
            "severity": SEVERITY[i],
        })

    # Sort descending by probability
    all_classes.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "diagnosis": CLASS_LABELS_CN[top_idx],
        "diagnosis_en": CLASS_LABELS_EN[top_idx],
        "top_class": top_idx,
        "confidence": round(top_prob, 4),
        "all_probabilities": all_classes,
        "severity": SEVERITY[top_idx],
        "urgency": URGENCY[top_idx],
    }
