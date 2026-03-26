# 子模块功能与特性文档

> 本文档汇总所有集成进 Life Monitoring 系统的子模块，包含功能描述、输入输出规范、模型特性及使用注意事项。

---

## 1. FaceAge — 面部生物年龄预测

**模块路径**：`module/FaceAge/`  
**模块入口**：`module/FaceAge/run.py`  
**原始入口**：`module/FaceAge/run_single_image.py`

### 功能描述

使用基于 FaceNet 架构的深度卷积神经网络，从单张面部照片估计人的**生物年龄**（biological age）。  
生物年龄与实际年龄的偏差可反映个体的整体健康状况和衰老速度。  
该模型在约 1 万名晚期癌症患者的临床影像数据上训练，专门针对 ≥40 岁人群优化。

### 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_path` | string | 面部图片路径，支持 JPG/PNG 等格式 |

### 输出（`outputs` 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `biological_age` | float | 估计生物年龄（岁），精度 0.1 岁 |
| `detection_confidence` | float | MTCNN 人脸检测置信度（0~1） |

### 模型特性

- **架构**：FaceNet 改进版（ResNet 类 + Inception 残差块，含 Lambda 缩放层）
- **人脸检测**：MTCNN 三阶段人脸检测器
- **预处理**：160×160 RGB 裁剪 → 逐图像标准化（均值/标准差归一化）
- **训练集**：临床晚期癌症患者影像（含生存期随访数据）
- **精度**：MAE ≈ 4.1 岁（≥40 岁人群）
- **模型文件**：`module/FaceAge/models/faceage_model.h5`（需单独下载）
  - 下载地址：https://github.com/AIM-Harvard/FaceAge/releases/download/v1/faceage_model.h5

### 依赖

- Python 3.11+，TensorFlow ≥2.16，tf-keras，mtcnn，scikit-image，Pillow

### 注意事项

- 模型使用 Python 3.6 字节码保存了 Lambda 层，`run.py` 内置 monkeypatch 修复兼容性问题
- 建议使用正面清晰照片，侧脸或模糊图像可能导致人脸检测失败
- 模型文件 `faceage_model.h5` 不含在仓库中，需手动下载

---

## 2. FaceTTD — 面部死亡时间预测

**模块路径**：`module/facettd/`  
**模块入口**：`module/facettd/run.py`  
**原始入口**：`module/facettd/app.py`（Gradio 界面）

### 功能描述

结合面部图像的 4096 维灰度像素特征与年龄信息，使用机器学习回归模型预测**距死亡的剩余时间**（TTD，Time To Death）及**预期寿命**。

### 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_path` | string | 面部图片路径 |
| `age` | int | 患者实际年龄（0~120 岁） |

### 输出（`outputs` 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `time_to_death_years` | float | 预测剩余寿命（年） |
| `life_expectancy_age` | float | 预期寿命（岁） = 当前年龄 + 剩余寿命 |
| `model_used` | string | 实际使用的模型文件名 |

### 模型特性

- **特征提取**：图像转为 64×64 灰度图，展平为 4096 维向量；年龄经 `StandardScaler` 标准化后拼接
- **输入维度**：4098（4096 图像 + 1 年龄 + 1 性别占位）

**三个可选模型**（按优先级选用）：

| 模型文件 | 类型 | 输入维度 | 说明 |
|----------|------|----------|------|
| `best_xgb.pkl` | XGBoost 回归 | 4436 | 含图像 + 元数据 + 死因编码特征 |
| `reg_xgb_ttd.pkl` | XGBoost 回归 | 4098 | 图像 + 年龄 |
| `reg_rf_ttd.pkl` | Random Forest 回归 | 4098 | 图像 + 年龄 |

- **年龄归一化**：`age_scaler.pkl`（训练时 `StandardScaler` 保存）

### 依赖

- Python 3.9+，numpy，Pillow，joblib，scikit-learn，xgboost

### 注意事项

- 模型文件（`.pkl`）需位于 `module/facettd/` 目录下
- 若 `age_scaler.pkl` 缺失，将使用原始年龄值作为 fallback（可能影响精度）
- 模型在特定临床数据集上训练，预测结果仅供研究参考

---

## 3. Parkinsons — 帕金森声学检测

**模块路径**：`module/Parkinsons/`  
**模块入口**：`module/Parkinsons/run.py`  
**核心实现**：`module/Parkinsons/detect.py`

### 功能描述

从 WAV 语音录音中提取 **22 个声学特征**，使用 XGBoost 分类器检测帕金森病风险。  
特征涵盖基频统计、声门微扰（Jitter/Shimmer）、谐噪比、以及非线性动力学特征。

### 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio_path` | string | WAV 格式语音文件路径（支持持续元音发声，如"啊"） |

### 输出（`outputs` 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `diagnosis` | string | `"正常"` 或 `"疑似帕金森"` |
| `prob_parkinson` | float | 帕金森概率（0~1） |
| `prob_healthy` | float | 正常概率（0~1） |
| `risk_level` | string | `"低风险"` / `"中等风险"` / `"高风险"` |
| `label` | int | 0=正常，1=帕金森 |

### 22 个声学特征

| 类别 | 特征名称 |
|------|----------|
| 基频 F0 | MDVP:Fo(Hz)、MDVP:Fhi(Hz)、MDVP:Flo(Hz) |
| 频率微扰 (Jitter) | MDVP:Jitter(%)、MDVP:Jitter(Abs)、MDVP:RAP、MDVP:PPQ、Jitter:DDP |
| 振幅微扰 (Shimmer) | MDVP:Shimmer、MDVP:Shimmer(dB)、Shimmer:APQ3、Shimmer:APQ5、MDVP:APQ、Shimmer:DDA |
| 谐噪比 | NHR、HNR |
| 非线性动力学 | RPDE（样本熵近似）、DFA（去趋势波动分析）、spread1、spread2、D2（关联维数）、PPE（基频周期熵） |

### 模型特性

- **分类器**：XGBoost（logloss 损失，random_state=42）
- **特征归一化**：MinMaxScaler 缩放到 [-1, 1]
- **训练集**：UCI Parkinson's Disease Dataset（`parkinsons.csv`，195 条样本，147 帕金森 + 48 正常）
- **训练策略**：每次调用动态训练（75% 训练，25% 测试）
- **特征提取工具**：Parselmouth（Praat Python API）、nolds（非线性动力学）

### 风险等级判定

| 帕金森概率 | 风险等级 |
|-----------|----------|
| > 75% | 高风险 |
| 50% ~ 75% | 中等风险 |
| < 50% | 低风险 |

### 依赖

- Python 3.9+，praat-parselmouth，nolds，xgboost，scikit-learn，pandas，numpy

### 注意事项

- 每次调用均重新训练模型（使用 `parkinsons.csv`），首次调用约需 3~10 秒
- 建议使用质量较好的 WAV 文件（44100 Hz，单声道），M4A 等格式需先转换
- 非线性特征（RPDE、DFA、D2）计算耗时较长，且依赖语音中有效有声帧数（建议 ≥3 秒）

---

## 4. Lung Cancer — 早期肺癌语音检测

**模块路径**：`module/Early_Stage_Lung_Cancer_Detection_from_Speech_Sounds/`  
**模块入口**：`run.py`（调用 `infer.py`）  
**训练流水线**：`main_codes/run_pipeline.py`  
**推断工具**：`infer.py`

### 功能描述

从语音录音中提取**时域统计特征、频域特征（FFT）和 MFCC** 特征，使用 **SVM + CNN + LSTM + Transformer** 四模型集成投票，检测早期肺癌语音信号。

### 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio_path` | string | WAV 格式语音文件路径 |

### 输出（`outputs` 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `final_verdict` | string | 综合判断结论 |
| `is_positive` | bool | 是否判断为肺癌阳性（≥2 个模型阳性） |
| `votes_positive` | int | 阳性投票模型数量（满分 4） |
| `total_models` | int | 总模型数（4） |
| `segments_analyzed` | int | 分析的音频片段数 |
| `model_results` | list | 各模型的详细预测结果 |

### `model_results` 结构

每个元素包含：

| 字段 | 说明 |
|------|------|
| `model` | 模型名称（SVM/CNN/LSTM/Transformer） |
| `mean_prob` | 平均肺癌概率 |
| `positive_segments` | 概率≥0.5 的片段数 |
| `total_segments` | 总片段数 |
| `verdict` | 单模型判断（`"疑似肺癌"` / `"未检出"`） |

### 特征提取（每个音频片段）

音频按 2048 采样点分段（hop=1024），每段提取 26 个特征：

| 类别 | 特征 |
|------|------|
| 时域统计（12 维） | MIN, MAX, MEAN, RMS, VAR, STD, POWER, PEAK, P2P, CREST FACTOR, SKEW, KURTOSIS |
| 频域统计（7 维） | MAX_f, SUM_f, MEAN_f, VAR_f, PEAK_f, SKEW_f, KURTOSIS_f |
| MFCC（13 维） | MFCC1 ~ MFCC13 |
| PCA 降维（3 维） | PCA1, PCA2, PCA3（模型训练时拟合的 PCA 变换） |

### 四个集成模型

| 模型 | 架构 | 说明 |
|------|------|------|
| SVM | RBF 核，C=10 | 基线模型，sklearn SVC |
| CNN | Conv1d(1→32) + FC(64→2) | 1D 卷积，PyTorch |
| LSTM | LSTM(hidden=64) + FC | 序列建模，PyTorch |
| Transformer | TransformerEncoder(d_model=F, nhead=1) + FC | 注意力机制，PyTorch |

**投票规则**：≥2 个模型判断阳性 → 综合判断为阳性（疑似肺癌）

### 训练流水线

需先运行一次完整训练：

```bash
python module/Early_Stage_Lung_Cancer_Detection_from_Speech_Sounds/main_codes/run_pipeline.py
```

训练完成后，以下文件会保存到 `saved_models/`：

- `svm_model.pkl`、`cnn_model.pt`、`lstm_model.pt`、`transformer_model.pt`
- `scaler.pkl`（StandardScaler）
- `selected_features.pkl`（FeatureWiz 选择的特征列表）
- `pca_transformer.pkl`、`pca_base_cols.pkl`（PCA 变换）

### 依赖

- Python 3.9+，torch，librosa，scipy，scikit-learn，pandas，numpy，featurewiz

### 注意事项

- **必须先训练**：推断前需确保 `saved_models/` 中存在所有模型文件
- 训练数据集：`dataset/feature_dataset.csv`（包含 CLASSES 列：0=健康，1=肺癌）
- 首次训练约需 5~30 分钟（取决于硬件，支持 MPS/CUDA/CPU）
- 音频分段处理，较短音频（<1 秒）可能导致片段数不足

---

## 5. SkinDisease — 皮肤病检测

**模块路径**：`module/SkinDisease/`
**模块入口**：`module/SkinDisease/run.py`
**核心实现**：`module/SkinDisease/predict.py`

### 功能描述

基于 EfficientNetV2B0 深度卷积神经网络，从面部或皮肤照片检测 **6 种常见皮肤病**，可辅助初步筛查与健康评估。模型在真实皮肤病数据集上训练，测试准确率 **95.6%**。

### 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_path` | string | 皮肤/面部图片路径（支持 JPG/PNG 等格式） |

### 输出（`outputs` 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `diagnosis` | string | 疾病名称（中文） |
| `diagnosis_en` | string | 疾病名称（英文） |
| `confidence` | float | 置信度（0~1） |
| `severity` | string | `"低风险"` / `"中等风险"` / `"高风险"` |
| `urgency` | string | 就医建议（中文） |
| `top_class` | int | 类别索引（0~5） |
| `all_probabilities` | list | 全部 6 类的概率详情 |

### 6 种皮肤病

| 类别 | 中文名 | 严重程度 |
|------|--------|----------|
| 0 rosacea | 酒糟鼻（玫瑰痤疮） | 中等风险 |
| 1 mila | 粟丘疹（白头） | 低风险 |
| 2 keratosis | 角化病（脂溢性角化病） | 低风险 |
| 3 eczema | 湿疹 | 中等风险 |
| 4 carcinoma | 皮肤癌（鳞状/基底细胞癌） | **高风险** |
| 5 acne | 痤疮 | 低风险 |

### 模型特性

- **架构**：EfficientNetV2B0（ImageNet 预训练，微调）
- **输入尺寸**：224×224×3
- **预处理**：`keras.applications.efficientnet_v2.preprocess_input`（像素缩放至 [-1, 1]）
- **模型来源**：[Tanishq77/skin-condition-classifier](https://huggingface.co/Tanishq77/skin-condition-classifier)（Hugging Face）
- **模型文件**：`module/SkinDisease/model/skin_model.keras`（需下载）
  - 下载地址（自动下载，首次调用时触发）：`huggingface_hub hf_hub_download(repo_id="Tanishq77/skin-condition-classifier", filename="skin_model.keras")`

### 依赖

- Python 3.11+，tensorflow-macos 2.16.1，keras ≥3.0，Pillow ≥10.0，numpy <2.0

### 注意事项

- **本模块仅供研究参考**，不能替代专业皮肤科医生的诊断
- 皮肤癌（carcinoma）类别置信度较高时，建议**尽快就医**
- 与 FaceAge/FaceTTD 模块可共用同一张面部照片，无需重复上传
- 首次调用约需 10-30 秒（模型加载 + 首次推理）

---

## 统一输出规范（UnifiedResult）

所有模块的 `run.py` 必须输出符合以下 JSON 格式的结果（写入 stdout）：

```json
{
  "module_id": "parkinsons",
  "module_name": "帕金森声学检测 (Parkinsons)",
  "status": "success",
  "inputs": {
    "audio": "voice.wav"
  },
  "outputs": {
    "diagnosis": "疑似帕金森",
    "prob_parkinson": 0.8231,
    "prob_healthy": 0.1769,
    "risk_level": "高风险",
    "label": 1
  },
  "summary": "帕金森检测结果：疑似帕金森（风险等级：高风险），帕金森概率 82.3%，正常概率 17.7%。",
  "error": null
}
```

**`status` 取值**：`"success"` | `"error"` | `"disabled"`

> 详见 `core/result_schema.py`（Pydantic `UnifiedResult` 模型）
