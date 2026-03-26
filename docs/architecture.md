# 主架构设计文档

## 系统概述

Life Monitoring 是一个**多模态健康分析系统**，集成多个小型 AI 模型，接收面部图像、语音录音和年龄等输入，各子模块独立运行并返回结构化分析结果，最终将所有结果汇总为一段**大模型提示词**，用于评估个体的预期生存期。

---

## 目录结构

```
Life-Monitoring/
├── module/                        # 子模块集合
│   ├── FaceAge/                   # 面部生物年龄预测
│   │   ├── run.py                 # 统一入口（stdin JSON → stdout JSON）
│   │   ├── run_single_image.py    # 原始推断脚本
│   │   └── models/faceage_model.h5
│   ├── facettd/                   # 面部死亡时间预测
│   │   ├── run.py
│   │   └── app.py                 # 原始 Gradio 界面
│   ├── Parkinsons/                # 帕金森声学检测
│   │   ├── run.py
│   │   ├── detect.py              # 核心检测逻辑
│   │   └── parkinsons.csv
│   ├── SkinDisease/               # 皮肤病检测（EfficientNetV2B0）
│   │   ├── run.py                 # 统一入口
│   │   ├── predict.py             # 核心推断逻辑
│   │   ├── model/                 # 模型文件（skin_model.keras）
│   │   └── .venv/                 # 隔离 Python 环境（TF 2.16.1）
│   └── Early_Stage_Lung_Cancer_Detection_from_Speech_Sounds/
│       ├── run.py                 # 统一入口
│       ├── infer.py               # 推断工具（从 run_pipeline.py 提取）
│       └── main_codes/run_pipeline.py  # 训练流水线
│
├── core/                          # 主架构核心层
│   ├── result_schema.py           # Pydantic UnifiedResult 模型
│   ├── module_runner.py           # subprocess 动态调用器
│   └── orchestrator.py            # 编排器（并发调用 + 渲染提示词）
│
├── config/                        # 配置文件
│   ├── modules.yaml               # 模块注册表（元数据、输入输出描述）
│   └── prompt_template.yaml       # 可编辑的 Jinja2 提示词模板
│
├── web/                           # Web 应用层
│   ├── app.py                     # FastAPI 后端
│   └── static/
│       ├── index.html             # 主页面
│       ├── style.css              # 样式
│       └── main.js                # 前端逻辑
│
├── docs/                          # 文档
│   ├── modules.md                 # 子模块功能说明
│   └── architecture.md            # 本文档
│
├── data/                          # 测试数据
│   ├── image/                     # 测试面部图片
│   └── sound/                     # 测试音频文件
│
├── uploads/                       # 运行时临时上传目录（自动创建）
└── requirements.txt               # Python 依赖
```

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         Web 浏览器                           │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │ 面部图片  │  │ 语音录音  │  │   年龄    │  │ 模块开关  │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └─────┬─────┘  │
│       └─────────────┴──────────────┴───────────────┘        │
│                          POST /api/analyze                    │
└─────────────────────────────┬───────────────────────────────┘
                               │ multipart/form-data
┌─────────────────────────────▼───────────────────────────────┐
│                      FastAPI 后端 (web/app.py)               │
│  • 保存上传文件到临时目录                                     │
│  • 调用 orchestrate()                                        │
│  • 返回 { results, prompt }                                   │
└─────────────────────────────┬───────────────────────────────┘
                               │
┌─────────────────────────────▼───────────────────────────────┐
│                   编排器 (core/orchestrator.py)              │
│  1. 对每个启用的模块，构建 inputs 字典                        │
│  2. ThreadPoolExecutor 并发调用 module_runner                 │
│  3. 收集所有 UnifiedResult                                    │
│  4. 使用 Jinja2 渲染 prompt_template.yaml                    │
└──────┬──────────┬──────────┬──────────────┬─────────────────┘
       │          │          │              │  (并发 subprocess)
       ▼          ▼          ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ FaceAge  │ │ FaceTTD  │ │Parkinson │ │LungCancer│
│ run.py   │ │ run.py   │ │  run.py  │ │  run.py  │
│          │ │          │ │          │ │          │
│stdin:JSON│ │stdin:JSON│ │stdin:JSON│ │stdin:JSON│
│stdout:   │ │stdout:   │ │stdout:   │ │stdout:   │
│  JSON    │ │  JSON    │ │  JSON    │ │  JSON    │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
       │          │          │              │
       └──────────┴──────────┴──────────────┘
                  UnifiedResult JSON
```

---

## 核心层设计

### 1. `core/result_schema.py` — 统一结果模型

所有模块的输出必须符合 `UnifiedResult` Pydantic 模型：

```python
class UnifiedResult(BaseModel):
    module_id:    str                 # "parkinsons"
    module_name:  str                 # "帕金森声学检测 (Parkinsons)"
    status:       str                 # "success" | "error" | "disabled"
    inputs:       dict[str, Any]      # 实际使用的输入（文件名、年龄等）
    outputs:      dict[str, Any]      # 模块核心输出（键值对）
    summary:      str                 # 人类可读的一句话摘要
    error:        Optional[str]       # 错误信息（status=error 时）
```

### 2. `core/module_runner.py` — 动态模块调用器

通过 `subprocess.run()` 动态调用任意模块的 `run.py`：

- **输入传递**：通过 `stdin` 传递 JSON（避免命令行参数长度限制和特殊字符问题）
- **输出解析**：从 `stdout` 解析最后一个 JSON 对象（跳过模块的 debug 打印）
- **隔离性**：每个模块在独立进程中运行，依赖冲突不影响主进程
- **超时控制**：默认 300 秒超时
- **环境继承**：传递 `PYTHONPATH=ROOT_DIR`，支持 `from core.result_schema import ...`

```python
run_module(module_id, inputs, python_exe=None) → UnifiedResult
```

### 3. `core/orchestrator.py` — 编排器

编排模块的并发执行和结果汇总：

- **并发执行**：`ThreadPoolExecutor(max_workers=4)` 并发调用所有启用的模块
- **输入路由**：自动判断模块所需输入类型（IMAGE_MODULES / AUDIO_MODULES）
- **提示词渲染**：使用 Jinja2 将结果填充到 `config/prompt_template.yaml` 模板中
- **模板持久化**：`save_prompt_template()` 将用户编辑的模板写回 YAML 文件

---

## API 设计

### `POST /api/analyze`

主分析接口，接收 `multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `age` | int | 是 | 患者实际年龄 |
| `enabled_modules` | string (JSON array) | 是 | 启用的模块 ID 列表 |
| `image` | file | 否 | 面部图片（FaceAge/FaceTTD 需要） |
| `audio` | file | 否 | 语音录音（Parkinsons/LungCancer 需要） |
| `prompt_override` | string | 否 | 临时提示词模板（不覆盖保存的模板） |

**响应** `200 OK`：

```json
{
  "results": {
    "faceage": {
      "module_id": "faceage",
      "module_name": "面部生物年龄预测 (FaceAge)",
      "status": "success",
      "inputs": { "image": "face.jpg" },
      "outputs": { "biological_age": 58.3, "detection_confidence": 0.997 },
      "summary": "面部生物年龄估计为 58.3 岁...",
      "error": null
    },
    "parkinsons": { ... }
  },
  "prompt": "你是一名经验丰富的医学专家，请基于以下多模态健康分析结果..."
}
```

### `GET /api/modules`

返回 `config/modules.yaml` 中的模块元数据列表，供前端渲染模块卡片。

### `GET /api/config/template`

返回当前提示词模板字符串。

### `PUT /api/config/template`

```json
{ "template": "Jinja2 模板字符串..." }
```

持久化用户编辑的模板到 `config/prompt_template.yaml`。

---

## 提示词模板系统

模板文件：`config/prompt_template.yaml`

- **格式**：Jinja2 模板语法，存储在 YAML 的 `template` 字段
- **变量**：`{{ age }}`、`{{ faceage.summary }}`、`{{ faceage.outputs.biological_age }}`、`{{ parkinsons.ok }}` 等（详见文件头部注释）
- **条件块**：使用 `{% if faceage.ok %}...{% endif %}` 仅在模块成功时输出对应段落
- **编辑方式**：
  1. 直接编辑 `config/prompt_template.yaml` 文件
  2. 通过 Web UI "编辑模板" 按钮在线编辑（自动调用 `PUT /api/config/template`）

---

## 子模块接入规范

如需新增子模块，遵循以下步骤：

1. **创建目录**：`module/<模块名>/`
2. **编写入口**：`module/<模块名>/run.py`
   - 从 `sys.stdin` 读取 JSON 输入
   - 将推断结果以 `UnifiedResult` JSON 格式写到 `sys.stdout`
3. **注册元数据**：在 `config/modules.yaml` 中添加模块描述（id、name、inputs、outputs）
4. **注册路由**：在 `core/module_runner.py` 的 `script_map` 中添加映射
5. **路由输入**：在 `core/orchestrator.py` 的 `IMAGE_MODULES` 或 `AUDIO_MODULES` 集合中声明所需输入类型
6. **更新模板**：在 `config/prompt_template.yaml` 中添加新模块的输出段落

---

## 启动方式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web 服务（开发模式）
uvicorn web.app:app --reload --port 8000

# 访问
open http://localhost:8000
```

**肺癌模块前置训练**（首次使用时运行一次）：

```bash
python module/Early_Stage_Lung_Cancer_Detection_from_Speech_Sounds/main_codes/run_pipeline.py
```

---

## 技术选型说明

| 组件 | 选型 | 原因 |
|------|------|------|
| Web 框架 | FastAPI | 原生支持异步、multipart、自动 OpenAPI 文档 |
| 数据验证 | Pydantic v2 | 类型安全、自动序列化/反序列化 |
| 模块调用 | subprocess | 完全进程隔离，避免各模块依赖冲突 |
| 并发 | ThreadPoolExecutor | subprocess 调用本质是 I/O 等待，线程池足够 |
| 模板引擎 | Jinja2 | 语法灵活、支持条件块，与 LLM Prompt 工程契合 |
| 前端 | 原生 HTML/CSS/JS | 无构建依赖，启动即用，易于二次修改 |
