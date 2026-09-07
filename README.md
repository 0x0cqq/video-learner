# Video Learner

将单个本地视频或 Bilibili 课时缓存转换成带截图、文字和来源时间戳的 Markdown，并通过 CLI 局部修订文字与配图，减少学习资料整理成本。

## 当前状态

P0 三条命令已实现：`inspect`、`convert`、`revise`。可将单视频转换为图文 Markdown，并局部修订文字、指定帧和裁剪，保留旧版本及非目标手改内容。

Windows 下 42 项离线测试、Ruff 和构建检查通过。PPT、编程、数学真实片段已完成转换与修订；Qwen 图文模型已完成约 100 分钟编程课（34 章）和约 45 分钟数学课（15 章）的完整转换。语音识别统一使用阿里云 Qwen，来源精确到音频切片区间。人工内容正确性和整理耗时对照仍待验收，详见[验证记录](docs/validation.md)。

## 文档入口

- [用户故事](docs/user-stories.md)：P0/P1 与验收标准。
- [技术选型](docs/technical-design.md)：媒体、转写、模型、证据和版本设计。
- [实现指南](docs/implementation-guide.md)：准备状态、M0–M5 与验证矩阵。
- [AGENTS.md](AGENTS.md)：参与本项目的协作规则。
- [使用指南](docs/usage.md)：安装、配置、转换、修订和错误处理。
- [验证记录](docs/validation.md)：实际检查、真实样本与质量限制。
- [实现笔记与 ADR](docs/implementation-notes.md)：开发过程、实验和技术选择原因。

## 首版范围

命令形成“检查素材 → 转换初稿 → 核对 → 局部修订 → 导出”闭环。P0 使用 Python 3.12、uv、Typer、PyAV、Qwen ASR，以及 DeepSeek / Qwen 多模态适配器，产物存为 Markdown、图片和 JSON。

自主补查、断点恢复和自然语言重新选图属于 P1。每次处理一个视频；产品范围不含问答、批量课程处理或学习管理。

## 安装与使用

在项目目录使用 PowerShell：

```powershell
uv sync --locked
uv run video-learner --help
uv run video-learner inspect "C:\Users\cqqqwq\Videos\bilibili\41301577497" --decode
uv run video-learner convert "C:\Users\cqqqwq\Videos\bilibili\41301577497" --start 00:45:00 --end 00:55:00 --profile programming --asr-secret "C:\projects\video-learner\secrets\aliyun.secret" --secret "C:\projects\video-learner\secrets\deepseek.secret" --output "C:\projects\video-learner\output\my-notes"
```

图文整理默认连接 DeepSeek `deepseek-v4-flash-vision-exp`，凭据用 `DEEPSEEK_API_KEY` 或 `--secret`。转换和文字修订也可用 `--provider qwen` 选择阿里云 `qwen3.8-flash`，默认开启思考和流式响应，凭据用 `DASHSCOPE_API_KEY` 或 `--secret`。Qwen ASR 使用同一阿里云环境变量或独立的 `--asr-secret`。不调用 OpenAI 服务，无需安装本地识别模型；指定同步字幕可跳过云端识别。完整示例见[使用指南](docs/usage.md)。

真实视频位于仓库外；`output/`、`artifacts/`、`secrets/` 和模型均不提交。每个成功版本的 `notes.md` 和 `assets/` 可一起复制；继续修订须保留完整工作目录。
