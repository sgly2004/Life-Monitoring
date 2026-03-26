# FaceTrack — 视频人脸切分模块

## 功能描述

基于 **facenet-pytorch**（InceptionResnetV1 + MTCNN）的人脸嵌入向量比对，从完整视频中自动切分出指定人物的片段序列。

## 技术方案

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 人脸检测 | MTCNN | 精准人脸检测 + 对齐裁剪，160×160 输入 |
| 人脸嵌入 | InceptionResnetV1（VGGFace2 预训练） | 512 维嵌入向量，cosine similarity 比对 |
| 设备支持 | MPS（Apple Silicon）/ CUDA / CPU | 自动检测，M1/M2/M3 Mac 可用 GPU 加速 |
| 视频 I/O | OpenCV | 逐帧读取，按间隔采样降算力 |

## 工作流程

```
参考人脸照片 → MTCNN 裁剪 → InceptionResnetV1 → 512维标准嵌入向量

视频逐帧 → MTCNN 检测人脸 → 提取嵌入向量
                              ↓
                    与参考嵌入计算 cosine similarity
                              ↓
                  相似度 > 阈值 → 标记为目标帧
                              ↓
              相邻目标帧合并为连续片段 → 输出
```

## 依赖安装

```bash
cd /Users/liuqiyuan/Documents/项目/Life-Monitoring/module/FaceTrack
pip install -r requirements.txt
```

## 调用方式

### 通过主系统 Web UI

在"人脸切分" Tab 页面，上传：
1. **目标人物面部照片**（参考）
2. **完整视频文件**

点击"开始人脸切分"即可。

### 通过命令行

```bash
cd /Users/liuqiyuan/Documents/项目/Life-Monitoring
python -c "
import json, sys
sys.path.insert(0, 'module/FaceTrack')
from face_tracker import track_faces_in_video

clips, total = track_faces_in_video(
    video_path='data/video/test.mp4',
    reference_face_path='data/image/ref.jpg',
    threshold=0.70,
    frame_interval=5,
)
print(json.dumps({'clips': clips, 'total_duration': total}, ensure_ascii=False, indent=2))
"
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.70 | cosine similarity 阈值，越高越严格 |
| `frame_interval` | int | 5 | 每隔 N 帧检测一次（降采样加速） |

## 输出格式

```json
{
  "module_id": "facetrack",
  "module_name": "视频人脸切分 (FaceTrack)",
  "status": "success",
  "inputs": {
    "video": "test.mp4",
    "reference_face": "ref.jpg",
    "threshold": 0.70
  },
  "outputs": {
    "clips": [
      {
        "start_time": 12.5,
        "end_time": 45.2,
        "duration": 32.7,
        "confidence": 0.93
      }
    ],
    "total_clips": 3,
    "total_duration": 128.4,
    "target_identity": "target_person"
  },
  "summary": "在视频中检测到 3 个目标人物片段，总时长 128.4 秒",
  "error": null
}
```
