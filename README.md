# Video Learner

将单个本地视频或 Bilibili 课时缓存转换成带截图、文字和来源时间戳的 Markdown，并通过 CLI 局部修订文字与配图，减少学习资料整理成本。

## 当前状态

P0 三条命令已实现：`inspect`、`convert`、`revise`。可将单视频转换为图文 Markdown，并局部修订文字、按时间替换完整截图，保留旧版本及非目标手改内容。

自动采样处理片段尾部的帧边界，Markdown 支持列表和引用内代码块，文字修订同步维护目标范围的待核对项。对应边界与离线验证见[技术设计](docs/technical-design.md)和[验证记录](docs/validation.md)。

`inspect` 分别报告媒体探测与解码结果，部分媒体探测失败时返回诊断和非零退出码；具体状态含义见[技术设计](docs/technical-design.md#2-导入与规范时间线)。

正文保留连贯解释与必要截图，来源和修订 ID 移至 `sources.md`，图注默认留空或简短提示。采样采用相邻画面比较、切换前留帧和周期补帧；音频在窗口末尾附近找低音量停顿，章节尽量对齐转写边界，保持实现简单。

`convert --jobs N` 可并发 ASR 请求（默认 1），图文整理保持串行，见[ASR 并发](docs/usage.md#asr-并发)。转换和修订默认显示阶段进度，转换阶段结束附实际处理数量与耗时，`--verbose` 查看详细事件，见[终端显示说明](docs/usage.md#终端进度与详细日志)。转换结束自动显示 Token meter 和 Estimated 费用，并保存 `usage.json`，包括重试用量、未知部分和所用单价。默认价格来自用户提供的 Qwen 报价及 DeepSeek 公开价格快照，可通过 TOML 覆盖；见[计费配置](docs/usage.md#token-meter-与估算费用)。

Windows 下离线测试、Ruff 和构建检查通过。新策略已用数学、编程各 4 分钟真实片段验证，并复验了图注精简、版本保护和独立复制；费用汇总也完成真实短片段验证。此前约 100 分钟编程课与约 45 分钟数学课的完整结果保留，本轮未重跑整课。语音识别默认使用阿里云 Qwen，可选本地 CPU/CUDA，来源精确到音频切片区间。测试数量、实际测量及人工验收限制统一见[验证记录](docs/validation.md)。

默认图文模型为 DeepSeek V4.1 Flash（`deepseek-flash`），转换以有界历史前缀争取缓存命中，实际用量已用四课短片段验证。可选本地 CPU/CUDA ASR 已在当前台式机完成一分钟样本 profiling；结果与质量限制见[性能分析](docs/performance.md#本地-asr-短片段测量)及[验证记录](docs/validation.md)。

## 文档入口

源码按 `common`、`workflows`、`media`、`providers`、`notes` 分组，CLI 入口保留在包根目录；目录职责见[代码组织](docs/implementation-guide.md#3-当前代码组织)。

- [用户故事](docs/user-stories.md)：P0/P1 与验收标准。
- [技术选型](docs/technical-design.md)：媒体、转写、模型、证据和版本设计。
- [实现指南](docs/implementation-guide.md)：准备状态、M0–M5 与验证矩阵。
- [AGENTS.md](AGENTS.md)：参与本项目的协作规则。
- [使用指南](docs/usage.md)：安装、配置、转换、修订和错误处理。
- [验证记录](docs/validation.md)：实际检查、真实样本与质量限制。
- [性能分析](docs/performance.md)：整课耗时归因、cProfile 结果和复现方法。
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

图文整理默认连接 DeepSeek `deepseek-flash`，凭据用 `DEEPSEEK_API_KEY` 或 `--secret`。转换和文字修订也可用 `--provider qwen` 选择阿里云 `qwen3.8-flash`，默认开启思考和流式响应，凭据用 `DASHSCOPE_API_KEY` 或 `--secret`。Qwen ASR 使用同一阿里云环境变量或独立的 `--asr-secret`。不调用 OpenAI 服务；云端路径无需本地权重，可选本地识别的安装见[使用指南](docs/usage.md#本地语音识别)。指定同步字幕可跳过云端识别。完整示例见[使用指南](docs/usage.md)。

真实视频位于仓库外；`output/`、`artifacts/`、`secrets/` 和模型均不提交。每个成功版本的 `notes.md` 和 `assets/` 可一起复制；继续修订须保留完整工作目录。
