"""
肺癌语音检测完整流水线（全 PyTorch 实现，避免 macOS 上 TensorFlow 线程问题）
SVM / CNN / LSTM / Transformer + 新音频推断
"""

import os, sys, pickle, warnings, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def log(msg=""):
    print(msg, flush=True)

import numpy as np
import pandas as pd
import librosa
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# ─── 路径配置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, 'dataset', 'feature_dataset.csv')
MODEL_DIR    = os.path.join(PROJECT_DIR, 'saved_models')
WAV_PATH     = os.path.normpath(os.path.join(PROJECT_DIR, '..', 'data', 'sound', '20260312_151645.wav'))
os.makedirs(MODEL_DIR, exist_ok=True)

log(f"数据集路径: {DATASET_PATH}")
log(f"模型保存目录: {MODEL_DIR}")
log(f"待检测音频:   {WAV_PATH}")
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
log(f"使用设备: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# 第一步：数据加载与预处理
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("【第一步】加载并预处理数据")
log("="*60)

data = pd.read_csv(DATASET_PATH)
log(f"数据集大小: {data.shape}")

y_all = data['CLASSES']
X_all = data.drop(columns=['CLASSES', 'Segment', 'Stage', 'PAT_ID'])
for col in ['KURTOSIS', 'KURTOSIS_f', 'SKEW', 'SKEW_f']:
    X_all[col] = X_all[col].astype(str).str.strip('[]')
    X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
X_all = X_all.dropna()
y_all = y_all.loc[X_all.index].reset_index(drop=True)
X_all = X_all.reset_index(drop=True)
log(f"清洗后: {X_all.shape}  健康(0)={sum(y_all==0)} 肺癌(1)={sum(y_all==1)}")

# ══════════════════════════════════════════════════════════════════════════════
# 第二步：FeatureWiz 特征选择
# ══════════════════════════════════════════════════════════════════════════════
log("\n【第二步】FeatureWiz 特征选择（corr_limit=0.95）...")
from featurewiz import FeatureWiz
fwiz = FeatureWiz(corr_limit=0.95, feature_engg='', category_encoders='',
                  dask_xgboost_flag=False, nrows=None, verbose=0)
result = fwiz.fit_transform(X_all, y_all)
X_sel = result[0] if isinstance(result, tuple) else result
selected_cols = list(X_sel.columns)
log(f"选择的特征: {len(selected_cols)} 个  →  {selected_cols}")

with open(os.path.join(MODEL_DIR, 'selected_features.pkl'), 'wb') as f:
    pickle.dump(selected_cols, f)

# PCA 拟合（供新音频使用）
pca_base_cols = [c for c in X_all.columns if c not in ['PCA1','PCA2','PCA3']]
pca_transformer = PCA(n_components=3).fit(X_all[pca_base_cols].fillna(0))
with open(os.path.join(MODEL_DIR, 'pca_transformer.pkl'), 'wb') as f:
    pickle.dump(pca_transformer, f)
with open(os.path.join(MODEL_DIR, 'pca_base_cols.pkl'), 'wb') as f:
    pickle.dump(pca_base_cols, f)

# 标准化 + 划分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all)
log(f"训练集: {X_train.shape}  测试集: {X_test.shape}")
input_dim = X_train.shape[1]

# 转为 PyTorch Tensor
def to_tensors(X, y):
    return (torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y.values, dtype=torch.long).to(device))

Xtr_t, ytr_t = to_tensors(X_train, y_train)
Xte_t, yte_t = to_tensors(X_test,  y_test)
train_ds = TensorDataset(Xtr_t, ytr_t)

# ══════════════════════════════════════════════════════════════════════════════
# 第三步：训练四个模型（纯 PyTorch + sklearn SVM）
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("【第三步】训练四个模型")
log("="*60)

def train_torch_model(model, train_ds, n_epochs=20, batch_size=256, lr=1e-3):
    """通用 PyTorch 训练函数，CrossEntropyLoss，打印 epoch 进度"""
    loader    = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for epoch in range(1, n_epochs+1):
        model.train()
        total_loss, n_correct, n_total = 0, 0, 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n_correct  += (logits.argmax(1) == yb).sum().item()
            n_total    += len(yb)
        log(f"   Epoch {epoch:2d}/{n_epochs}  "
            f"loss={total_loss/n_total:.4f}  "
            f"train_acc={n_correct/n_total:.4f}  "
            f"elapsed={time.time()-t0:.0f}s")
    return model

def eval_model(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
    return preds, probs

# ─── 1. SVM ──────────────────────────────────────────────────────────────────
log("\n▶ 1/4  SVM (rbf, C=10)...")
t0 = time.time()
svm_clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_clf.fit(X_train, y_train)
log(f"   完成 ({time.time()-t0:.0f}s)")
with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm_clf, f)
y_pred_svm  = svm_clf.predict(X_test)
y_probs_svm = svm_clf.predict_proba(X_test)[:, 1]
log(f"   SVM 准确率: {accuracy_score(y_test, y_pred_svm):.4f}  ✓")

# ─── 2. CNN（PyTorch Conv1d）────────────────────────────────────────────────
log("\n▶ 2/4  CNN (PyTorch Conv1d, 20 epochs)...")

class CNNClassifier(nn.Module):
    def __init__(self, in_features, n_classes=2):
        super().__init__()
        # Conv1d(1, 32, k=3, padding=1) keeps sequence length = in_features
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
        out = self.conv(x.unsqueeze(1))          # (B, 32, F)
        return self.fc(out.view(out.size(0), -1)) # (B, 32*F)

cnn_model = CNNClassifier(input_dim).to(device)
cnn_model = train_torch_model(cnn_model, train_ds, n_epochs=20)
torch.save(cnn_model.state_dict(), os.path.join(MODEL_DIR, 'cnn_model.pt'))
y_pred_cnn, y_probs_cnn = eval_model(cnn_model, Xte_t, yte_t)
log(f"   CNN 准确率: {accuracy_score(y_test, y_pred_cnn):.4f}  ✓")

# ─── 3. LSTM（PyTorch）──────────────────────────────────────────────────────
log("\n▶ 3/4  LSTM (PyTorch, 20 epochs)...")

class LSTMClassifier(nn.Module):
    def __init__(self, in_features, hidden=64, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden, n_classes)
    def forward(self, x):
        # x: (B, F) → (B, F, 1)
        out, _ = self.lstm(x.unsqueeze(2))
        return self.fc(out[:, -1, :])

lstm_model = LSTMClassifier(input_dim).to(device)
lstm_model = train_torch_model(lstm_model, train_ds, n_epochs=20)
torch.save(lstm_model.state_dict(), os.path.join(MODEL_DIR, 'lstm_model.pt'))
y_pred_lstm, y_probs_lstm = eval_model(lstm_model, Xte_t, yte_t)
log(f"   LSTM 准确率: {accuracy_score(y_test, y_pred_lstm):.4f}  ✓")

# ─── 4. Transformer（PyTorch）───────────────────────────────────────────────
log("\n▶ 4/4  Transformer (PyTorch, 20 epochs)...")

class TransformerClassifier(nn.Module):
    def __init__(self, in_features, n_classes=2):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_features, nhead=1,
                                       dim_feedforward=128, batch_first=True),
            num_layers=1)
        self.fc = nn.Linear(in_features, n_classes)
    def forward(self, x):
        # x: (B, F) → (B, 1, F) → TransformerEncoder → (B, F)
        out = self.encoder(x.unsqueeze(1)).squeeze(1)
        return self.fc(out)

tf_model = TransformerClassifier(input_dim).to(device)
tf_model = train_torch_model(tf_model, train_ds, n_epochs=20)
torch.save(tf_model.state_dict(), os.path.join(MODEL_DIR, 'transformer_model.pt'))
y_pred_tf, y_probs_tf = eval_model(tf_model, Xte_t, yte_t)
log(f"   Transformer 准确率: {accuracy_score(y_test, y_pred_tf):.4f}  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# 第四步：详细评估 + ROC 曲线
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("【第四步】详细评估指标（测试集 20%）")
log("="*60)

results = {
    'SVM':         (y_pred_svm,  y_probs_svm),
    'CNN':         (y_pred_cnn,  y_probs_cnn),
    'LSTM':        (y_pred_lstm, y_probs_lstm),
    'Transformer': (y_pred_tf,   y_probs_tf),
}
colors = {'SVM':'steelblue','CNN':'tomato','LSTM':'seagreen','Transformer':'mediumpurple'}
fpr_d, tpr_d, auc_d = {}, {}, {}

for name, (y_pred, y_prob) in results.items():
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fpr_d[name], tpr_d[name], auc_d[name] = fpr, tpr, roc_auc
    log(f"\n── {name} ──────────────────────────────────────────")
    log(f"  准确率: {acc:.4f}  精确率: {prec:.4f}  召回率: {rec:.4f}  F1: {f1:.4f}  AUC: {roc_auc:.4f}")
    log(f"  混淆矩阵 (行=真实, 列=预测):\n{confusion_matrix(y_test, y_pred)}")

# ── 5 折 CV（SVM）────────────────────────────────────────────────────────────
log("\n【第五步】5 折交叉验证（SVM）")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr, te) in enumerate(kf.split(X_scaled, y_all)):
    clf_cv = SVC(kernel='rbf', C=10, gamma='scale')
    clf_cv.fit(X_scaled[tr], y_all.iloc[tr])
    sc = accuracy_score(y_all.iloc[te], clf_cv.predict(X_scaled[te]))
    cv_scores.append(sc)
    log(f"  Fold {fold+1}: {sc:.4f}")
log(f"  SVM 5-折 平均准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ── ROC 图 ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
for name in results:
    plt.plot(fpr_d[name], tpr_d[name], color=colors[name], lw=2.5,
             label=f'{name} (AUC={auc_d[name]:.4f})')
plt.plot([0,1],[0,1],'--',color='gray',lw=1)
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curves – Lung Cancer Detection\n(SVM / CNN / LSTM / Transformer)', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
roc_path = os.path.join(PROJECT_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=150)
log(f"\nROC 曲线已保存: {roc_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 第六步：从 WAV 提取特征
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("【第六步】从音频文件提取特征")
log("="*60)

def extract_segment_features(seg, sr, n_mfcc=13):
    seg = seg.astype(np.float64)
    rms = float(np.sqrt(np.mean(seg**2)))
    feat = [float(np.min(seg)), float(np.max(seg)), float(np.mean(seg)),
            rms, float(np.var(seg)), float(np.std(seg)),
            float(np.mean(seg**2)), float(np.max(np.abs(seg))),
            float(np.max(seg) - np.min(seg)),
            float(np.max(np.abs(seg)) / (rms + 1e-10)),
            float(scipy.stats.skew(seg)), float(scipy.stats.kurtosis(seg))]
    fft_amp = np.abs(np.fft.rfft(seg)) + 1e-10
    feat += [float(np.max(fft_amp)), float(np.sum(fft_amp)), float(np.mean(fft_amp)),
             float(np.var(fft_amp)), float(np.max(fft_amp)),
             float(scipy.stats.skew(fft_amp)), float(scipy.stats.kurtosis(fft_amp))]
    mfccs = librosa.feature.mfcc(y=seg.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    feat += mfccs.mean(axis=1).tolist()
    return feat

def extract_features_from_wav(wav_path, segment_len=2048, hop_len=1024):
    log(f"  加载: {wav_path}")
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    log(f"  采样率:{sr} Hz  时长:{len(audio)/sr:.2f}s  总样本:{len(audio)}")
    col_names = (['MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P',
                  'CREST FACTOR','SKEW','KURTOSIS']
                 + ['MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f']
                 + [f'MFCC{i}' for i in range(1, 14)])
    rows = [extract_segment_features(audio[s:s+segment_len], sr)
            for s in range(0, len(audio) - segment_len + 1, hop_len)]
    df = pd.DataFrame(rows, columns=col_names)

    with open(os.path.join(MODEL_DIR, 'pca_transformer.pkl'), 'rb') as f:
        pca_tf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'pca_base_cols.pkl'), 'rb') as f:
        pca_base = pickle.load(f)
    avail  = [c for c in pca_base if c in df.columns]
    pca_in = df[avail].fillna(0).values
    if pca_in.shape[1] < pca_tf.n_features_in_:
        pca_in = np.hstack([pca_in, np.zeros((len(pca_in), pca_tf.n_features_in_ - pca_in.shape[1]))])
    pca_out = pca_tf.transform(pca_in)
    df['PCA1'], df['PCA2'], df['PCA3'] = pca_out[:,0], pca_out[:,1], pca_out[:,2]
    log(f"  提取片段数: {len(df)}")
    return df

if not os.path.exists(WAV_PATH):
    log(f"  !! 未找到音频: {WAV_PATH}")
    sys.exit(0)

df_wav = extract_features_from_wav(WAV_PATH)

with open(os.path.join(MODEL_DIR, 'selected_features.pkl'), 'rb') as f:
    sel_cols = pickle.load(f)
for col in sel_cols:
    if col not in df_wav.columns:
        df_wav[col] = 0.0
X_wav_s = scaler.transform(df_wav[sel_cols].fillna(0).values)

# ══════════════════════════════════════════════════════════════════════════════
# 第七步：推断
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("【第七步】对音频片段进行肺癌检测推断")
log("="*60)

probs_all = {}

# SVM
with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'rb') as f:
    svm_l = pickle.load(f)
probs_all['SVM'] = svm_l.predict_proba(X_wav_s)[:, 1]

# CNN / LSTM / Transformer（PyTorch）
X_wav_t = torch.tensor(X_wav_s, dtype=torch.float32).to(device)

def load_and_predict(ModelClass, weight_path, X_tensor, *args):
    m = ModelClass(*args).to(device)
    m.load_state_dict(torch.load(weight_path, map_location=device))
    m.eval()
    with torch.no_grad():
        logits = m(X_tensor)
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

probs_all['CNN']         = load_and_predict(CNNClassifier,         os.path.join(MODEL_DIR,'cnn_model.pt'),         X_wav_t, input_dim)
probs_all['LSTM']        = load_and_predict(LSTMClassifier,        os.path.join(MODEL_DIR,'lstm_model.pt'),        X_wav_t, input_dim)
probs_all['Transformer'] = load_and_predict(TransformerClassifier, os.path.join(MODEL_DIR,'transformer_model.pt'), X_wav_t, input_dim)

log(f"\n音频文件: {os.path.basename(WAV_PATH)}")
log(f"分析片段数: {len(df_wav)}\n")
log(f"{'模型':<14} {'平均肺癌概率':>12} {'阳性片段':>14} {'判断':>10}")
log("-"*56)
votes = []
for name, probs in probs_all.items():
    mp    = float(np.mean(probs))
    pos   = int(np.sum(probs >= 0.5))
    ratio = pos / len(probs) * 100
    judge = "⚠ 疑似肺癌" if mp >= 0.5 else "✓ 未检出"
    votes.append(mp >= 0.5)
    log(f"{name:<14} {mp:>11.4f}  {pos:>5}/{len(probs)} ({ratio:4.1f}%)  {judge}")
log("-"*56)
yes   = sum(votes)
final = "⚠ 多数模型阳性，建议就医确认！" if yes >= 2 else "✓ 多数模型未检出肺癌信号"
log(f"\n【综合投票】{yes}/4 个模型判断为阳性")
log(f"【最终判断】{final}")

# 概率分布图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'各模型对音频片段的肺癌概率分布\n({os.path.basename(WAV_PATH)})', fontsize=14)
for ax, (name, probs) in zip(axes.flatten(), probs_all.items()):
    ax.hist(probs, bins=30, color=colors[name], alpha=0.75, edgecolor='black')
    ax.axvline(0.5,               color='red',  linestyle='--', lw=1.5, label='阈值 0.5')
    ax.axvline(float(np.mean(probs)), color='navy', linestyle='-',  lw=1.5, label=f'均值 {np.mean(probs):.3f}')
    ax.set_title(f'{name}  (均值={np.mean(probs):.3f})', fontsize=12)
    ax.set_xlabel('肺癌概率'); ax.set_ylabel('片段数'); ax.legend(fontsize=9)
plt.tight_layout()
dist_path = os.path.join(PROJECT_DIR, 'prediction_distribution.png')
plt.savefig(dist_path, dpi=150)
log(f"\n预测分布图已保存: {dist_path}")

log("\n" + "="*60)
log("全部步骤完成！")
log("="*60)
