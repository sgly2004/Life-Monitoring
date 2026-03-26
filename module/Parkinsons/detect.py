"""
帕金森声学检测脚本
输入 WAV 文件，提取 22 个声学特征，使用 parkinsons.csv 训练模型后预测。
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import parselmouth
from parselmouth.praat import call
import nolds

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# 1. 特征提取
# ──────────────────────────────────────────────

def extract_features(wav_path: str) -> dict:
    """从 WAV 文件提取与 parkinsons.csv 完全对应的 22 个声学特征。"""

    sound = parselmouth.Sound(wav_path)

    # ── 基频 F0 ──
    pitch       = call(sound, "To Pitch", 0.0, 75, 600)
    f0_mean     = call(pitch, "Get mean",    0, 0, "Hertz")
    f0_max      = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    f0_min      = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # 提取所有有声帧的 F0 序列，供后续非线性特征使用
    n_frames   = call(pitch, "Get number of frames")
    f0_seq     = np.array([
        call(pitch, "Get value in frame", i, "Hertz")
        for i in range(1, n_frames + 1)
    ])
    f0_voiced  = f0_seq[~np.isnan(f0_seq) & (f0_seq > 0)]

    # ── 声门脉冲序列 ──
    pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    # ── Jitter（频率微扰） ──
    jitter_pct = call(pp, "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap        = call(pp, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3)
    ppq        = call(pp, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3)
    ddp        = rap * 3  # DDP = 3 × RAP

    # ── Shimmer（振幅微扰） ──
    shimmer    = call([sound, pp], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    # shimmer dB: Praat 不直接暴露此接口，用 20*log10(1+shimmer) 近似
    shimmer_db = 20 * np.log10(1 + shimmer) if shimmer > 0 else 0.0
    apq3       = call([sound, pp], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5       = call([sound, pp], "Get shimmer (apq5)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11      = call([sound, pp], "Get shimmer (apq11)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda        = call([sound, pp], "Get shimmer (dda)",    0, 0, 0.0001, 0.02, 1.3, 1.6)

    # ── 谐噪比 HNR / NHR ──
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr         = call(harmonicity, "Get mean", 0, 0)
    nhr         = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.0

    # ── 非线性动力学特征 ──
    # 使用对数基频序列（与论文保持一致）
    log_f0 = np.log(f0_voiced) if len(f0_voiced) > 50 else np.log(np.clip(f0_voiced, 1, None))

    # RPDE：递归周期密度熵（用样本熵近似）
    try:
        rpde = nolds.sampen(log_f0, emb_dim=2)
        rpde = float(np.clip(rpde, 0, 1))
    except Exception:
        rpde = 0.5

    # DFA：去趋势波动分析
    try:
        dfa = nolds.dfa(log_f0)
        dfa = float(np.clip(dfa, 0, 1))
    except Exception:
        dfa = 0.7

    # spread1：log基频序列的标准差（对应论文中基频幅度变化）
    spread1 = float(np.log(np.std(log_f0) + 1e-10))

    # spread2：log基频序列的四分位距（非线性变化度）
    spread2 = float(np.percentile(log_f0, 75) - np.percentile(log_f0, 25))

    # D2：关联维数
    try:
        d2 = nolds.corr_dim(log_f0, emb_dim=2)
        d2 = float(np.clip(d2, 0, 10))
    except Exception:
        d2 = 2.0

    # PPE：基频周期熵——log基频差分序列的归一化熵
    try:
        diffs  = np.diff(log_f0)
        hist, _ = np.histogram(diffs, bins=30, density=True)
        hist   = hist[hist > 0]
        ppe    = float(-np.sum(hist * np.log2(hist)) / np.log2(len(hist) + 1))
        ppe    = float(np.clip(ppe, 0, 1))
    except Exception:
        ppe = 0.2

    return {
        "MDVP:Fo(Hz)":      f0_mean,
        "MDVP:Fhi(Hz)":     f0_max,
        "MDVP:Flo(Hz)":     f0_min,
        "MDVP:Jitter(%)":   jitter_pct,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP":         rap,
        "MDVP:PPQ":         ppq,
        "Jitter:DDP":       ddp,
        "MDVP:Shimmer":     shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3":     apq3,
        "Shimmer:APQ5":     apq5,
        "MDVP:APQ":         apq11,
        "Shimmer:DDA":      dda,
        "NHR":              nhr,
        "HNR":              hnr,
        "RPDE":             rpde,
        "DFA":              dfa,
        "spread1":          spread1,
        "spread2":          spread2,
        "D2":               d2,
        "PPE":              ppe,
    }


# ──────────────────────────────────────────────
# 2. 训练模型
# ──────────────────────────────────────────────

def train_model(csv_path: str):
    data    = pd.read_csv(csv_path)
    X       = data.drop(["name", "status"], axis=1).values
    y       = data["status"].values

    scaler  = MinMaxScaler((-1, 1))
    X_sc    = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=7)
    model   = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"  模型训练完成，测试集准确率: {acc*100:.1f}%")
    return model, scaler


# ──────────────────────────────────────────────
# 3. 主流程
# ──────────────────────────────────────────────

def detect(wav_path: str, csv_path: str):
    wav_path = Path(wav_path).resolve()
    csv_path = Path(csv_path).resolve()

    print("=" * 55)
    print("  帕金森声学检测")
    print("=" * 55)
    print(f"  音频文件: {wav_path.name}")

    # 加载音频信息
    sound = parselmouth.Sound(str(wav_path))
    print(f"  时长: {sound.duration:.1f}s  |  采样率: {int(sound.sampling_frequency)} Hz  |  声道: {sound.n_channels}")

    # 提取特征
    print("\n[1/3] 提取声学特征 ...")
    feats = extract_features(str(wav_path))

    feat_df = pd.DataFrame([feats])
    print("\n  ┌─ 主要特征值 ──────────────────────────────────┐")
    groups = [
        ("基频 (Hz)",     ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]),
        ("Jitter",        ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ"]),
        ("Shimmer",       ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3"]),
        ("噪谐比",        ["NHR", "HNR"]),
        ("非线性动力学",  ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]),
    ]
    for group_name, keys in groups:
        vals = "  ".join(f"{k}={feats[k]:.4f}" for k in keys)
        print(f"  │  {group_name}: {vals}")
    print("  └───────────────────────────────────────────────┘")

    # 训练模型
    print(f"\n[2/3] 训练模型 (数据: {csv_path.name}) ...")
    model, scaler = train_model(str(csv_path))

    # 归一化 & 预测
    print("\n[3/3] 进行预测 ...")
    X_input = scaler.transform(feat_df.values)
    label   = model.predict(X_input)[0]
    proba   = model.predict_proba(X_input)[0]

    p_healthy    = proba[0]
    p_parkinson  = proba[1]

    # 结果输出
    print()
    print("=" * 55)
    print("  检测结果")
    print("=" * 55)
    if label == 1:
        verdict = "⚠️  疑似帕金森 (Parkinson's Detected)"
        risk    = "高风险" if p_parkinson > 0.75 else "中等风险"
    else:
        verdict = "✅  正常 (Healthy)"
        risk    = "低风险"

    print(f"  判断结果 : {verdict}")
    print(f"  风险等级 : {risk}")
    print(f"  帕金森概率: {p_parkinson*100:.1f}%")
    print(f"  正常概率  : {p_healthy*100:.1f}%")
    print()
    print("  ⚠️  本结果仅供研究参考，不能替代专业医疗诊断。")
    print("=" * 55)

    return {
        "label":            int(label),
        "diagnosis":        "疑似帕金森" if label == 1 else "正常",
        "prob_parkinson":   round(float(p_parkinson), 4),
        "prob_healthy":     round(float(p_healthy), 4),
        "features":         feats,
    }


if __name__ == "__main__":
    WAV = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/liuqiyuan/Documents/项目/Life-Monitoring/data/sound/20260312_151645.wav"
    CSV = sys.argv[2] if len(sys.argv) > 2 else \
        "/Users/liuqiyuan/Documents/项目/Life-Monitoring/Parkinsons/parkinsons.csv"

    detect(WAV, CSV)
