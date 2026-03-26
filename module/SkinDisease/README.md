# SkinDisease — 皮肤病检测模块

基于 **EfficientNetV2B0** 深度卷积神经网络的皮肤病图像分类模块，可识别 6 种常见皮肤病变。

---

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | EfficientNetV2B0（ImageNet 预训练，微调） |
| 输入尺寸 | 224 × 224 × 3（RGB） |
| 输出 | 6 类 Softmax 概率分布 |
| 测试准确率 | 95.6% |
| 模型来源 | [Tanishq77/skin-condition-classifier](https://huggingface.co/Tanishq77/skin-condition-classifier)（Hugging Face） |

### 6 种皮肤病分类

| 类别索引 | 英文名 | 中文名 | 严重程度 |
|---------|--------|--------|----------|
| 0 | rosacea | 酒糟鼻（玫瑰痤疮） | 中等风险 |
| 1 | mila | 粟丘疹（白头） | 低风险 |
| 2 | keratosis | 角化病（脂溢性角化病） | 低风险 |
| 3 | eczema | 湿疹 | 中等风险 |
| 4 | carcinoma | 皮肤癌（鳞状/基底细胞癌） | **高风险** |
| 5 | acne | 痤疮 | 低风险 |

---

## 模型文件

模型文件需单独下载（不包含在仓库中）。

### 自动下载（首次使用时自动）

运行 `run.py` 时，如检测到模型文件缺失，将自动从 Hugging Face 下载。

### 手动下载

```bash
cd module/SkinDisease/model
python ../../.venv/bin/python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('Tanishq77/skin-condition-classifier', 'skin_model.keras')
print('Downloaded to:', path)
"
```

---

## 使用方法

### 作为独立脚本

```bash
echo '{"image_path": "/path/to/skin_image.jpg"}' | \
  .venv/bin/python run.py
```

**输出示例：**

```json
{
  "module_id": "skin_disease",
  "module_name": "皮肤病检测 (SkinDisease)",
  "status": "success",
  "inputs": {"image": "skin_image.jpg"},
  "outputs": {
    "diagnosis": "湿疹",
    "diagnosis_en": "eczema",
    "confidence": 0.8231,
    "severity": "中等风险",
    "urgency": "建议皮肤科就诊，日常护理与药物治疗可缓解",
    "top_class": 3,
    "all_probabilities": [...]
  },
  "summary": "皮肤病检测结果：湿疹，置信度 82.3%，严重程度：中等风险。前三候选：湿疹 82.3%、痤疮 10.2%、酒糟鼻 5.1%。",
  "error": null
}
```

### 通过 Web 应用

通过 `POST /api/analyze` 接口提交图片时勾选「皮肤病检测」模块即可。

---

## 依赖环境

| 包 | 版本 |
|----|------|
| tensorflow-macos | 2.16.1 |
| keras | ≥3.0 |
| Pillow | ≥10.0 |
| numpy | <2.0 |

环境已内置于 `.venv/` 隔离虚拟环境中，**无需额外安装**。

---

## 输入说明

- 支持任何 PIL 可读的图像格式（JPG、PNG 等）
- 建议使用清晰的皮肤病灶照片或面部近照
- 图像会自动缩放至 224×224，无需手动预处理
- **无需手动裁剪人脸**，模型直接对整张图片进行分类

---

## 注意事项

- 本模块仅供研究参考，不能替代专业皮肤科医生的诊断
- 皮肤癌（carcinoma）类别置信度较高时，建议**尽快就医**
- 首次调用约需 10-30 秒（模型加载 + 首次推理）
- 与 FaceAge 模块可共用同一张面部照片，无需重复上传
