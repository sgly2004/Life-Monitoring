# VoiceTrack — 音频声纹切分模块

## 功能描述

基于 **pyannote-audio** 说话人分割 + **resemblyzer** 声纹嵌入比对，从完整音频中自动提取指定说话人的所有片段。

## 技术方案

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 说话人分割 | pyannote-audio (3.1) | 预训练 diarization 模型，自动标注每段音频的说话人 |
| 声纹嵌入 | resemblyzer (VoiceEncoder) | 256 维语音嵌入，vggish 预训练模型 |
| 设备支持 | MPS（Apple Silicon）/ CUDA / CPU | 自动检测 |
| 音频 I/O | librosa + resemblyzer.audio | 自动采样率转换（16kHz） |

## 工作流程

```
参考语音 → resemblyzer VoiceEncoder → 256维标准声纹向量

完整音频 → pyannote-audio 说话人分割 → 各片段时间戳
                                        ↓
                        提取各片段声纹嵌入向量
                                        ↓
                              与参考嵌入计算 cosine similarity
                                        ↓
                          相似度 > 阈值 → 保留该片段
                                        ↓
                      相邻片段合并（间隔<1.5s）→ 输出
```

## 依赖安装

```bash
cd /Users/liuqiyuan/Documents/项目/Life-Monitoring/module/VoiceTrack
pip install -r requirements.txt

# pyannote 需要 Hugging Face token（免费）
# 1. 注册 Hugging Face 账号：https://huggingface.co/
# 2. 申请 pyannote 模型访问权限：
#    https://huggingface.co/pyannote/speaker-diarization-3.1
#    https://huggingface.co/pyannote/embedding-3.1
# 3. 创建 access token 并设置环境变量
export HF_TOKEN="your_huggingface_token_here"
```

## 调用方式

### 通过主系统 Web UI

在"声纹切分" Tab 页面，上传：
1. **目标说话人语音**（参考，清晰单一说话人）
2. **完整音频文件**

点击"开始声纹切分"即可。

### 通过命令行

```bash
cd /Users/liuqiyuan/Documents/项目/Life-Monitoring
export HF_TOKEN="your_token"
python -c "
import json, sys
sys.path.insert(0, 'module/VoiceTrack')
from voice_tracker import track_voice_in_audio

clips, total = track_voice_in_audio(
    audio_path='data/sound/test.wav',
    reference_audio_path='data/sound/ref.wav',
    threshold=0.75,
)
print(json.dumps({'clips': clips, 'total_duration': total}, ensure_ascii=False, indent=2))
"
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.75 | cosine similarity 阈值，越高越严格 |

## 输出格式

```json
{
  "module_id": "voicetrack",
  "module_name": "音频声纹切分 (VoiceTrack)",
  "status": "success",
  "inputs": {
    "audio": "meeting.wav",
    "reference_audio": "speaker_ref.wav",
    "threshold": 0.75
  },
  "outputs": {
    "clips": [
      {
        "start_time": 5.3,
        "end_time": 22.1,
        "duration": 16.8,
        "confidence": 0.89
      }
    ],
    "total_clips": 4,
    "total_duration": 45.20,
    "target_speaker": "target_speaker"
  },
  "summary": "在音频中检测到 4 个目标说话人片段，总时长 45.2 秒",
  "error": null
}
```

## 备选方案（无 pyannote）

如果 pyannote 模型访问受限，`voice_tracker.py` 内置了基于能量检测的 fallback 逻辑：
- 自动将音频切分为 5 秒一段
- 过滤静音片段（rms < 0.005）
- 仍使用 resemblyzer 进行声纹比对

## 注意事项

- 参考音频建议使用 **单人清晰语音**，时长至少 1 秒
- pyannote-audio 在 CPU 上较慢（5 分钟音频约需 1~3 分钟），M1/M2 Mac 建议用 MPS
- 支持格式：WAV、MP3、M4A、FLAC 等（librosa 自动转换）
