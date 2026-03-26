"""
FaceAge single-image inference script
Compatible with Python 3.11 + TF 2.20 + tf-keras (Keras 2 API)

Usage:
    python run_single_image.py path/to/photo.jpg
    python run_single_image.py   # defaults to data/image/2012-11-14～2026-3-12.png
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys
import argparse
import numpy as np
import PIL.Image
from skimage.io import imread

# ── Fix Lambda layer Python 3.6 bytecode incompatibility ──────────────────────
# All Lambda layers in this model compute: inputs[0] + inputs[1] * scale
# We monkeypatch func_load to return a proper function when marshal.loads fails.
import tf_keras.src.utils.generic_utils as _gu

_original_func_load = _gu.func_load

def _patched_func_load(code, globs=None, locs=None, closure=None):
    try:
        return _original_func_load(code, globs, locs, closure)
    except EOFError:
        # Fallback: return the ScaleSum lambda that all blocks use
        # Arguments (scale) are injected via the Layer's 'arguments' config
        return lambda inputs, scale=0.2: inputs[0] + inputs[1] * scale

_gu.func_load = _patched_func_load
# ──────────────────────────────────────────────────────────────────────────────

import tf_keras as keras
import mtcnn

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "faceage_model.h5")
DEFAULT_IMAGE   = os.path.join(BASE_DIR, "..", "data", "image", "2012-11-14～2026-3-12.png")


def detect_face(image_path):
    img = imread(image_path)
    # Convert to RGB (handles RGBA, grayscale, etc.)
    img = np.array(PIL.Image.fromarray(img).convert('RGB'))
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(img)
    if not results:
        raise RuntimeError("No face detected in the image.")
    return results[0], img


def crop_and_preprocess(img, mtcnn_result):
    x1, y1, w, h = mtcnn_result['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h

    face = img[y1:y2, x1:x2]

    # resize to model input (160×160 RGB)
    face_pil = PIL.Image.fromarray(np.uint8(face)).convert('RGB')
    face_resized = np.asarray(face_pil.resize((160, 160)))

    # per-image standardization
    mean, std = face_resized.mean(), face_resized.std()
    face_norm = (face_resized - mean) / std

    return face_norm.reshape(1, 160, 160, 3), (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser(description="FaceAge — estimate biological age from a face photo")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                        help="Path to input image (.jpg / .png / .jpeg). "
                             "Defaults to the bundled sample photo.")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("  Download: https://github.com/AIM-Harvard/FaceAge/releases/download/v1/faceage_model.h5")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        sys.exit(1)

    print(f"Image : {image_path}")
    print(f"Model : {MODEL_PATH}")
    print()

    # Stage 1: face detection
    print("Stage 1 — Detecting face (MTCNN)...")
    mtcnn_result, img = detect_face(image_path)
    confidence = mtcnn_result.get('confidence', 0)
    print(f"  Detected  (confidence: {confidence:.3f})")
    print(f"  Bbox      : {mtcnn_result['box']}")
    print()

    # Stage 2: crop + preprocess
    face_input, (x1, y1, x2, y2) = crop_and_preprocess(img, mtcnn_result)
    print(f"  Crop      : x=[{x1},{x2}]  y=[{y1},{y2}]  → 160×160 RGB")
    print()

    # Stage 3: load model and predict
    print("Stage 2 — Loading FaceAge model...")
    model = keras.models.load_model(MODEL_PATH)
    print("  Loaded.")
    print()

    print("Stage 3 — Running age estimation...")
    faceage = float(np.squeeze(model.predict(face_input, verbose=0)))
    print()
    print("=" * 40)
    print(f"  FaceAge : {faceage:.2f} years")
    print("=" * 40)
    print()
    print("FaceAge = biological age (not appearance).")
    print("Model optimized for age ≥ 40; MAE ≈ 4.1 yrs.")


if __name__ == '__main__':
    main()
