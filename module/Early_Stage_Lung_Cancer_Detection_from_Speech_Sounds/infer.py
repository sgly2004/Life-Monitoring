"""
Lung Cancer inference utilities — extracted from run_pipeline.py.
Provides model class definitions and a standalone infer() function.
Models must be pre-trained via main_codes/run_pipeline.py first.
"""
from __future__ import annotations
import os
import pickle
import warnings

import numpy as np
import scipy.stats
import librosa
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ── Model definitions (must match run_pipeline.py exactly) ────────────────────

class CNNClassifier(nn.Module):
    def __init__(self, in_features, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * in_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self._in = in_features

    def forward(self, x):
        out = self.conv(x.unsqueeze(1))
        return self.fc(out.view(out.size(0), -1))


class LSTMClassifier(nn.Module):
    def __init__(self, in_features, hidden=64, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(2))
        return self.fc(out[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self, in_features, n_classes=2):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_features, nhead=1, dim_feedforward=128, batch_first=True
            ),
            num_layers=1,
        )
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        out = self.encoder(x.unsqueeze(1)).squeeze(1)
        return self.fc(out)


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_segment_features(seg: np.ndarray, sr: int, n_mfcc: int = 13) -> list:
    seg = seg.astype(np.float64)
    rms = float(np.sqrt(np.mean(seg ** 2)))
    feat = [
        float(np.min(seg)), float(np.max(seg)), float(np.mean(seg)),
        rms, float(np.var(seg)), float(np.std(seg)),
        float(np.mean(seg ** 2)), float(np.max(np.abs(seg))),
        float(np.max(seg) - np.min(seg)),
        float(np.max(np.abs(seg)) / (rms + 1e-10)),
        float(scipy.stats.skew(seg)), float(scipy.stats.kurtosis(seg)),
    ]
    fft_amp = np.abs(np.fft.rfft(seg)) + 1e-10
    feat += [
        float(np.max(fft_amp)), float(np.sum(fft_amp)), float(np.mean(fft_amp)),
        float(np.var(fft_amp)), float(np.max(fft_amp)),
        float(scipy.stats.skew(fft_amp)), float(scipy.stats.kurtosis(fft_amp)),
    ]
    mfccs = librosa.feature.mfcc(y=seg.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    feat += mfccs.mean(axis=1).tolist()
    return feat


def _extract_features_from_wav(wav_path: str, segment_len: int = 2048, hop_len: int = 1024):
    import pandas as pd

    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    col_names = (
        ["MIN", "MAX", "MEAN", "RMS", "VAR", "STD", "POWER", "PEAK", "P2P",
         "CREST FACTOR", "SKEW", "KURTOSIS"]
        + ["MAX_f", "SUM_f", "MEAN_f", "VAR_f", "PEAK_f", "SKEW_f", "KURTOSIS_f"]
        + [f"MFCC{i}" for i in range(1, 14)]
    )
    rows = [
        _extract_segment_features(audio[s: s + segment_len], sr)
        for s in range(0, len(audio) - segment_len + 1, hop_len)
    ]
    df = pd.DataFrame(rows, columns=col_names)

    with open(os.path.join(MODEL_DIR, "pca_transformer.pkl"), "rb") as f:
        pca_tf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "pca_base_cols.pkl"), "rb") as f:
        pca_base = pickle.load(f)

    avail = [c for c in pca_base if c in df.columns]
    pca_in = df[avail].fillna(0).values
    if pca_in.shape[1] < pca_tf.n_features_in_:
        pca_in = np.hstack(
            [pca_in, np.zeros((len(pca_in), pca_tf.n_features_in_ - pca_in.shape[1]))]
        )
    pca_out = pca_tf.transform(pca_in)
    df["PCA1"], df["PCA2"], df["PCA3"] = pca_out[:, 0], pca_out[:, 1], pca_out[:, 2]
    return df


# ── Main inference function ────────────────────────────────────────────────────

def infer(wav_path: str) -> dict:
    """
    Run lung-cancer inference on a WAV file.
    Returns a dict with keys: final_verdict, votes_positive, total_models,
    model_results (list), segments_analyzed.
    Raises FileNotFoundError if models are missing.
    """
    required = [
        "svm_model.pkl", "cnn_model.pt", "lstm_model.pt",
        "transformer_model.pt", "scaler.pkl",
        "selected_features.pkl", "pca_transformer.pkl", "pca_base_cols.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"缺少训练好的模型文件：{missing}。\n"
            f"请先运行训练流水线：\n"
            f"  python module/Early_Stage_Lung_Cancer_Detection_from_Speech_Sounds/main_codes/run_pipeline.py"
        )

    import pandas as pd

    df_wav = _extract_features_from_wav(wav_path)

    with open(os.path.join(MODEL_DIR, "selected_features.pkl"), "rb") as f:
        sel_cols = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    for col in sel_cols:
        if col not in df_wav.columns:
            df_wav[col] = 0.0

    X_wav_s = scaler.transform(df_wav[sel_cols].fillna(0).values)
    input_dim = X_wav_s.shape[1]
    X_wav_t = torch.tensor(X_wav_s, dtype=torch.float32).to(DEVICE)

    probs_all: dict[str, np.ndarray] = {}

    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)
    probs_all["SVM"] = svm_model.predict_proba(X_wav_s)[:, 1]

    def _load_and_predict(ModelClass, weight_path, *args):
        m = ModelClass(*args).to(DEVICE)
        m.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        m.eval()
        with torch.no_grad():
            logits = m(X_wav_t)
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    probs_all["CNN"] = _load_and_predict(
        CNNClassifier, os.path.join(MODEL_DIR, "cnn_model.pt"), input_dim
    )
    probs_all["LSTM"] = _load_and_predict(
        LSTMClassifier, os.path.join(MODEL_DIR, "lstm_model.pt"), input_dim
    )
    probs_all["Transformer"] = _load_and_predict(
        TransformerClassifier, os.path.join(MODEL_DIR, "transformer_model.pt"), input_dim
    )

    model_results = []
    votes_positive = 0
    for name, probs in probs_all.items():
        mean_prob = float(np.mean(probs))
        positive_segs = int(np.sum(probs >= 0.5))
        is_positive = mean_prob >= 0.5
        if is_positive:
            votes_positive += 1
        model_results.append(
            {
                "model": name,
                "mean_prob": round(mean_prob, 4),
                "positive_segments": positive_segs,
                "total_segments": len(probs),
                "verdict": "疑似肺癌" if is_positive else "未检出",
            }
        )

    final_positive = votes_positive >= 2
    return {
        "final_verdict": "疑似肺癌（建议就医确认）" if final_positive else "多数模型未检出肺癌信号",
        "is_positive": final_positive,
        "votes_positive": votes_positive,
        "total_models": len(probs_all),
        "model_results": model_results,
        "segments_analyzed": len(df_wav),
    }
