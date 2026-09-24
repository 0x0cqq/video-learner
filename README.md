# Video Learner

将单个本地视频或 Bilibili 课时缓存转换成带截图、文字和来源时间戳的讲义，通过 CLI 局部修订文字与配图，导出 Markdown、HTML 和 PDF。

## 当前状态

P0 的 `inspect`、`convert`、`revise`、`export` 已实现。转换默认保存初稿 r001，再依次执行图文对应、讲义内容两轮独立复审，生成 `revisions/r002/notes.md`；CLI 返回复审版路径。可配置复审顺序、自定义规则和只报告模式，见[独立复审](docs/usage.md#独立复审)。局部修改保留旧版本及非目标手改内容。自动复审仍有漏检和误改，关键内容需要对照原课核对。语音识别默认使用阿里云 Qwen，可显式选择本地 CPU/CUDA；图文整理默认使用 DeepSeek，也可选择 Qwen。

新生成的转换和文字修订会保存模型调用前的证据输入与已采用草稿，可只读核对固定模型结果下的生成过程。当前行为与数据边界见[技术设计](docs/technical-design.md)，运行方法见[使用指南](docs/usage.md)，已执行验证及内容质量限制见[历史验证记录](docs/validation.md)。

HTML/PDF 阅读导出支持中文括号、引号边界的双星号加粗；Markdown 保留当前原文。外部编辑器的写法与格式支持见[阅读导出说明](docs/usage.md#导出阅读副本)。

## 文档入口

- [用户故事](docs/user-stories.md)：P0/P1 与验收标准。
- [技术设计](docs/technical-design.md)：架构、转换数据流、证据和版本契约。
- [实现指南](docs/implementation-guide.md)：源码职责、M0–M5 验收条件与验证矩阵。
- [AGENTS.md](AGENTS.md)：参与本项目的协作规则。
- [使用指南](docs/usage.md)：安装、配置、转换、修订、导出和错误处理。
- [性能分析工具](docs/performance.md)：性能测量与证据回放的使用方法。
- [历史验证记录](docs/validation.md)：实际检查、真实样本与验收限制。
- [历史内容质量记录](docs/content-quality.md)：整课阅读发现与校订结果。
- [实现笔记与 ADR](docs/implementation-notes.md)：开发过程、实验和技术选择原因。

## 首版范围

命令形成“检查素材 → 转换初稿 → 核对 → 局部修订 → 导出”闭环。P0 使用 Python 3.12、uv、Typer、PyAV、Qwen ASR，以及 DeepSeek / Qwen 多模态适配器，可编辑版本存为 Markdown、图片和 JSON，阅读副本可导出离线 HTML 或分页 PDF。

自主补查、断点恢复和自然语言重新选图属于 P1。每次处理一个视频；产品范围不含问答、批量课程处理或学习管理。

## 安装与使用

在项目目录使用 PowerShell。示例视频位于项目外的 `D:\media\lesson.mp4`；转换前按[使用指南](docs/usage.md#配置)准备两份凭据文件：

```powershell
uv sync --locked
uv run video-learner --help
uv run video-learner inspect "D:\media\lesson.mp4" --decode
uv run video-learner convert "D:\media\lesson.mp4" --asr-secret ".\secrets\aliyun.secret" --secret ".\secrets\deepseek.secret" --output ".\output\my-notes"
```

图文整理默认连接 DeepSeek `deepseek-flash`，凭据用 `DEEPSEEK_API_KEY` 或 `--secret`。转换和文字修订也可用 `--provider qwen` 选择阿里云 `qwen3.8-flash`，默认开启思考和流式响应，凭据用 `DASHSCOPE_API_KEY` 或 `--secret`。Qwen ASR 使用同一阿里云环境变量或独立的 `--asr-secret`。不调用 OpenAI 服务；云端路径无需本地权重，可选本地识别的安装见[使用指南](docs/usage.md#本地语音识别)。指定同步字幕可跳过云端识别。完整示例见[使用指南](docs/usage.md)。

导出现有讲义无需模型或原视频；各格式共享文档快照，按阅读方式分别排版：

```powershell
uv run video-learner export ".\output\my-notes" --base r002 --format html --output ".\output\my-notes-html"
uv run --extra pdf video-learner export ".\output\my-notes" --base r002 --format pdf --output ".\output\my-notes-pdf"
```

可重复传入 `--format` 一次导出多个格式，详见[导出阅读副本](docs/usage.md#导出阅读副本)。

真实视频位于仓库外；`output/`、`artifacts/`、`secrets/` 和模型均不提交。每个成功版本的 `notes.md` 和 `assets/` 可一起复制；继续修订须保留完整工作目录。
