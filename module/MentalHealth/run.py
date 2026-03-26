#!/usr/bin/env python3
"""
MentalHealth — Facial Expression Recognition for Mental Health Detection

Input  (stdin JSON): {"image_path": "...", "age": ...}
Output (stdout JSON): UnifiedResult schema
"""
import sys
import json
import os
import warnings
import math

warnings.filterwarnings("ignore")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(MODULE_DIR, "model")

# FER2013 7 emotion labels
EMOTION_LABELS = ["生气", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性"]
EMOTION_LABELS_EN = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Mental health score mapping: dominant emotion → base score adjustment
# Based on the original paper's mental health scoring methodology
EMOTION_MENTAL_SCORE = {
    "开心":    15,
    "惊讶":    10,
    "中性":     5,
    "恐惧":    -5,
    "悲伤":   -10,
    "生气":   -10,
    "厌恶":   -15,
}

# Mental health risk classification
MENTAL_RISK_CN = {
    "低风险":    "低风险",
    "中等风险":  "中等风险",
    "高风险":    "高风险",
}


RESULT_BASE = {
    "module_id": "mental_health",
    "module_name": "心理状态检测 (MentalHealth)",
    "status": "error",
    "inputs": {},
    "outputs": {},
    "summary": "",
    "error": None,
}


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _detect_and_crop_face(image_path):
    """Detect face in image and return cropped RGB face array."""
    import numpy as np
    import PIL.Image
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
    except ImportError:
        detector = None

    img = PIL.Image.open(image_path).convert("RGB")
    img_rgb = np.array(img)

    if detector is not None:
        detections = detector.detect_faces(img_rgb)
        if detections:
            x1, y1, w, h = detections[0]["box"]
            x1, y1 = abs(x1), abs(y1)
            face = img_rgb[y1: y1 + h, x1: x1 + w]
            face_pil = PIL.Image.fromarray(face.astype("uint8"))
        else:
            face_pil = img
    else:
        face_pil = img

    return face_pil.resize((224, 224))


def _build_model(num_classes=7):
    """Build and load the Swin Transformer model."""
    import torch
    import timm

    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def _compute_mental_health_score(emotion_probs: list[float], dominant_emotion: str) -> float:
    """
    Compute a mental health score (0-100) based on emotion probabilities.
    Formula: base 50 + emotion adjustment from dominant + entropy bonus

    emotion_probs: list of 7 probabilities (one per emotion class)
    dominant_emotion: Chinese label of the dominant emotion
    """
    import numpy as np

    max_prob = max(emotion_probs)
    dominant_adjustment = EMOTION_MENTAL_SCORE.get(dominant_emotion, 0)

    # Entropy bonus: more uniform distribution → slightly lower confidence
    # High entropy = more mixed emotions = potentially less stable mental state
    probs = np.array(emotion_probs)
    probs = probs[probs > 1e-8]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(7)  # uniform distribution
    entropy_ratio = entropy / max_entropy  # 0=confident, 1=confused

    entropy_bonus = (entropy_ratio - 1) * 5  # -5 to 0

    score = 50 + dominant_adjustment + entropy_bonus
    score = max(0.0, min(100.0, score))
    return round(score, 1)


def _classify_risk(score: float) -> tuple[str, str]:
    """Classify mental health risk from score (0-100)."""
    if score >= 70:
        return "低风险", "心理健康状况良好"
    elif score >= 55:
        return "中等风险", "存在轻度情绪波动，建议关注"
    elif score >= 40:
        return "高风险", "可能存在负面情绪累积，建议进一步评估"
    else:
        return "高风险", "情绪状态偏负面，建议寻求专业支持"


def _map_to_mental_state(emotion_probs: list[float], dominant_idx: int) -> dict:
    """Map emotion analysis to mental health state description."""
    import numpy as np

    probs = np.array(emotion_probs)

    happy_idx = EMOTION_LABELS.index("开心")
    sad_idx = EMOTION_LABELS.index("悲伤")
    fear_idx = EMOTION_LABELS.index("恐惧")
    angry_idx = EMOTION_LABELS.index("生气")

    positive = probs[happy_idx]
    negative = probs[[sad_idx, fear_idx, angry_idx]].sum()

    if probs[happy_idx] > 0.4:
        state = "积极乐观"
        description = "情绪整体积极，可能处于良好的心理状态"
    elif probs[sad_idx] > 0.35:
        state = "情绪低落"
        description = "悲伤情绪较为明显，可能需要关注心理状态"
    elif probs[fear_idx] > 0.3:
        state = "焦虑不安"
        description = "恐惧/焦虑情绪明显，可能存在压力或担忧"
    elif probs[angry_idx] > 0.35:
        state = "情绪激动"
        description = "愤怒情绪较明显，可能存在不满或压力"
    elif negative > positive + 0.2:
        state = "负面情绪占优"
        description = "整体情绪偏负面，建议关注心理健康"
    else:
        state = "情绪平稳"
        description = "情绪表现较为中性或混合"

    return {"state": state, "description": description}


def main():
    data = json.load(sys.stdin)
    image_path = data.get("image_path", "")
    age = data.get("age")

    result = dict(RESULT_BASE)
    result["inputs"] = {"image": os.path.basename(image_path), "age": age}

    try:
        import torch
        import numpy as np
        import PIL.Image

        model_path = os.path.join(MODEL_DIR, "best_model.pth")
        class_map_path = os.path.join(MODEL_DIR, "class_mapping.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在：{model_path}\n"
                "请先运行 train_model.py 在 CUDA 服务器上完成模型训练。\n"
                "训练完成后模型将保存到 MentalHealth/model/best_model.pth"
            )

        # Load class mapping
        if os.path.exists(class_map_path):
            with open(class_map_path, encoding="utf-8") as f:
                class_map = json.load(f)
        else:
            class_map = {i: label for i, label in enumerate(EMOTION_LABELS)}

        # Load model
        model = _build_model(num_classes=7)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Detect and preprocess face
        face_pil = _detect_and_crop_face(image_path)
        transform = _build_transform()
        face_tensor = transform(face_pil).unsqueeze(0)

        # Predict
        with torch.no_grad():
            logits = model(face_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        # Get dominant emotion
        dominant_idx = int(np.argmax(probs))
        dominant_emotion = class_map.get(str(dominant_idx), EMOTION_LABELS[dominant_idx])
        dominant_prob = float(probs[dominant_idx])

        # Emotion probabilities dict
        emotion_dict = {}
        for i, label in enumerate(EMOTION_LABELS):
            emotion_dict[label] = round(float(probs[i]), 4)

        # Compute mental health score
        mh_score = _compute_mental_health_score(probs.tolist(), dominant_emotion)
        risk_level, risk_suggestion = _classify_risk(mh_score)
        mental_state = _map_to_mental_state(probs.tolist(), dominant_idx)

        # Build result
        result["status"] = "success"
        result["outputs"] = {
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_en": EMOTION_LABELS_EN[dominant_idx],
            "dominant_probability": round(dominant_prob, 4),
            "all_emotion_probabilities": emotion_dict,
            "mental_health_score": mh_score,
            "mental_health_risk": risk_level,
            "mental_state": mental_state["state"],
            "mental_state_description": mental_state["description"],
            "age": age,
        }
        result["summary"] = (
            f"心理状态检测结果：主导情绪为「{dominant_emotion}」（置信度 {dominant_prob * 100:.1f}），"
            f"心理健康评分 {mh_score}/100（{risk_level}），"
            f"情绪状态：{mental_state['state']}。"
        )

    except FileNotFoundError:
        result["status"] = "error"
        result["error"] = f"模型文件不存在，请先训练模型"
        result["summary"] = f"MentalHealth 模块需要训练模型后才能使用"
        raise

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["summary"] = f"MentalHealth 模块运行出错：{exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
