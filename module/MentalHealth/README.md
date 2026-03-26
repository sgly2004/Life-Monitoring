# 心理状态检测 (MentalHealth)

基于面部表情识别（FER）的心理健康检测模块，使用 **Swin Transformer** 深度学习模型，分析人脸图像中的 7 种基本情绪（生气、厌恶、恐惧、开心、悲伤、惊讶、中性），并据此评估心理健康状态。

情绪分类 → 心理健康评分（0-100） → 风险等级 + 情绪状态描述

---

## 模型架构

- **主干网络**: Swin Transformer Base（`swin_base_patch4_window7_224`）
- **分类头**: Linear(1024 → 512) → ReLU → Dropout(0.6) → Linear(512 → 7)
- **预训练**: ImageNet 预训练权重
- **参考论文**: [Mujiyanto et al., ETASR 2024](https://doi.org/10.48084/etasr.9139)

## 情绪标签

| 索引 | 英文 | 中文 | 心理健康评分调整 |
|------|------|------|-----------------|
| 0 | angry | 生气 | -10 |
| 1 | disgust | 厌恶 | -15 |
| 2 | fear | 恐惧 | -5 |
| 3 | happy | 开心 | +15 |
| 4 | sad | 悲伤 | -10 |
| 5 | surprise | 惊讶 | +10 |
| 6 | neutral | 中性 | +5 |

## 心理健康评分公式

```
score = 50 + 情绪调整分 + 熵修正分
```

- **情绪调整分**: 主导情绪对应的调整值（见上表）
- **熵修正分**: 基于情绪分布熵的修正（-5 ~ 0），分布越均匀（混合情绪）分数略低

### 风险等级

| 评分范围 | 风险等级 | 建议 |
|----------|---------|------|
| 70-100 | 低风险 | 心理健康状况良好 |
| 55-69 | 中等风险 | 存在轻度情绪波动，建议关注 |
| 40-54 | 高风险 | 可能存在负面情绪累积，建议进一步评估 |
| 0-39 | 高风险 | 情绪状态偏负面，建议寻求专业支持 |

---

## 目录结构

```
module/MentalHealth/
├── model/                    # 训练后的模型文件（需手动放置）
│   ├── best_model.pth        # Swin Transformer 模型权重
│   └── class_mapping.json    # 类别映射
├── run.py                    # 推理入口（注册到 app 的模块）
├── train_model.py            # 训练脚本（仅在 CUDA 服务器运行）
└── requirements.txt          # Python 依赖
```

---

## 环境配置（仅需在 CUDA 服务器上操作一次）

```bash
cd module/MentalHealth
python -m venv venv_mh
source venv_mh/bin/activate
pip install -r requirements.txt
```

---

## 数据集准备

### 方式一：从 Kaggle 下载 FER2013

```bash
pip install kaggle
mkdir -p ~/.kaggle
# 将 kaggle.json 放入 ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
mkdir -p fer2013
tar -xvzf challenges-in-representation-learning-facial-expression-recognition-challenge.tar.gz -C fer2013
```

### 方式二：使用其他面部表情数据集

只要满足以下目录结构即可：

```
FER2013_processed/
├── train/
│   ├── angry/   disgust/   fear/   happy/   sad/   surprise/   neutral/
│   └── (每类一个文件夹，放入对应图片)
├── val/
│   └── (同上)
└── test/
    └── (同上)
```

或者只需一个 `train/` 目录（8:2 自动划分为训练集和验证集）。

---

## 训练模型（在 CUDA 服务器上）

```bash
cd module/MentalHealth
source venv_mh/bin/activate

# 标准训练
python train_model.py --epochs 50 --batch_size 32 --lr 1e-4

# 恢复中断的训练
python train_model.py --epochs 50 --batch_size 32 --lr 1e-4 \
    --checkpoint model/best_model.pth

# 指定数据集路径（如果数据集不在模块目录下）
python train_model.py --data_root /path/to/your/dataset --epochs 50
```

训练完成后，模型保存到 `model/best_model.pth`。

---

## 模型下载（可选）

如果原作者提供了预训练模型，可从以下链接下载：

```
https://drive.google.com/file/d/1-BzdfiqVkGgR3Qpptdsv0j3s94dWRy0l/view
```

下载后放置到 `module/MentalHealth/model/` 目录下。

---

## 推理使用

推理使用 `run.py`，由项目主 app 的 orchestrator 自动调用，**无需 CUDA**（模型在 CPU 上推理）。

输入：
```json
{"image_path": "/path/to/face.jpg", "age": 35}
```

输出（UnifiedResult）：
```json
{
  "module_id": "mental_health",
  "module_name": "心理状态检测 (MentalHealth)",
  "status": "success",
  "inputs": {"image": "face.jpg", "age": 35},
  "outputs": {
    "dominant_emotion": "开心",
    "dominant_emotion_en": "happy",
    "dominant_probability": 0.6234,
    "all_emotion_probabilities": {
      "生气": 0.02, "厌恶": 0.01, "恐惧": 0.03,
      "开心": 0.62, "悲伤": 0.05, "惊讶": 0.15, "中性": 0.12
    },
    "mental_health_score": 78.3,
    "mental_health_risk": "低风险",
    "mental_state": "积极乐观",
    "mental_state_description": "情绪整体积极，可能处于良好的心理状态",
    "age": 35
  },
  "summary": "心理状态检测结果：主导情绪为「开心」（置信度 62.3%），心理健康评分 78.3/100（低风险），情绪状态：积极乐观。"
}
```

---

## 注意事项

1. **训练必须在 CUDA 环境下进行**，推理可以在 CPU 上完成
2. 推理时会先尝试使用 MTCNN 检测人脸区域并裁剪；若 MTCNN 未安装，则直接使用整张图
3. 心理健康评分仅供参考，不能替代专业心理评估
4. 如果 MTCNN 检测不到人脸，推理结果可能不准确，请确保上传清晰正面人脸照片
