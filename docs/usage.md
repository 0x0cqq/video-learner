# 使用指南

## 安装

项目要求 Python 3.12 和 uv，优先验证 Windows。在项目根目录运行 `uv sync --locked`，然后使用 `uv run video-learner`。普通素材检查不加载 ASR 模型或读取 AI 凭据。

## 配置

图文供应商通过 `--provider deepseek|qwen` 或 TOML 的 `provider` 选择。默认 DeepSeek，固定连接 `https://api.deepseek.com`，模型为 `deepseek-v4-flash-vision-exp`；Qwen 固定连接 `https://dashscope.aliyuncs.com/compatible-mode/v1`，模型为 `qwen3.8-flash`。不自动切换服务，`openai` 包仅作为这两家服务的兼容客户端。

DeepSeek 凭据从 `DEEPSEEK_API_KEY` 读取，Qwen 从 `DASHSCOPE_API_KEY` 读取；也可使用 `--secret` 指定所选供应商的密钥文件（仅包含密钥的 UTF-8 文本）。不要将真实密钥写入 TOML、命令行参数或文档。仓库忽略 `secrets/`。

`--config` 指定 TOML；优先级为命令行 > TOML > 默认值。可参考 [config.example.toml](../config.example.toml)。明确要求额外解释时使用 `--instruction` 并加 `--allow-ai-additions`；生成内容会单独标为 AI 补充解释。

## 检查素材

```powershell
uv run video-learner inspect "C:\Users\cqqqwq\Videos\bilibili\41301577497" --json --decode
```

`--decode` 抽查前中后音视频，`--full` 顺序验证全部音视频解码。元数据下载完成、媒体打开、片段解码与全片验证分别显示。支持普通单视频文件及已知 Bilibili 单课时缓存，不递归处理多个课时，不下载在线视频。

## 转换

```powershell
uv run video-learner convert "C:\Users\cqqqwq\Videos\bilibili\41301577497" --start 00:45:00 --end 00:55:00 --profile programming --secret "C:\projects\video-learner\secrets\deepseek.secret" --output "C:\projects\video-learner\output\sample"
```

时间可用秒数或 `HH:MM:SS`，最多六位小数，按 `[start,end)` 处理。省略范围转换整课。输出必须是与源目录不重叠的新目录。

`--profile` 支持 `programming`、`math`、`mixed`；`--instruction` 设置组织要求。`--crop x,y,width,height` 使用原帧像素指定关注区域；默认保留全帧。固定采样和去重可能遗漏短暂画面，模型会从每章有界候选图中选图，请核对中间步骤。

`--subtitle` 可指定 UTF-8 SRT/VTT 替代 ASR。字幕需时间合法、覆盖当前范围；覆盖比例写入工作记录，同步仍需人工核对。发现但未显式选择的字幕不会自动覆盖语音识别。

默认使用阿里云 Qwen ASR，按原视频时间提取 16 kHz 单声道音频并发送云端识别；不安装或下载本地识别模型。

生成的根目录对应 r001：`notes.md`、`assets/`、`review.md`、`source.json`、`transcript.jsonl`、`notes.json`。`.work/` 保存本机来源、指纹、原帧、音频窗口、生成快照、版本清单与日志。

## 云端语音识别

转换默认使用 Qwen；ASR 与图文整理分别读取独立凭据：

```powershell
uv run video-learner convert "C:\Users\cqqqwq\Videos\bilibili\1272065375" --start 00:10:00 --end 00:20:00 --profile math --asr-secret "C:\projects\video-learner\secrets\aliyun.secret" --secret "C:\projects\video-learner\secrets\deepseek.secret" --output "C:\projects\video-learner\output\my-qwen-notes"
```

Qwen 凭据也可用 `DASHSCOPE_API_KEY`，与 DeepSeek 凭据独立。端点固定阿里云 DashScope，模型默认 `qwen3-asr-flash`。音频切片以 Base64 发送；默认每片 30 秒、最多 500 次调用，TOML 字段为 `asr_window_seconds`、`asr_max_calls`。窗口越短，回看区间越小、请求越多。

该兼容接口不返回句级时间戳；来源明确标注为真实音频切片区间，切片边界可能截断词句。10 分钟数学音频实测转写约 32 秒，不是所有课程的固定性能。认证、调用上限和未完成输出会明确失败。识别设置进入提取指纹，修订时复用基线证据。历史本地识别产物仍可修订，无需重新转写。旧配置中的本地模型/模式参数及 CLI 的 ASR 切换选项已不再接受。

## Qwen 图文理解

语音识别和图文整理均使用阿里云的示例：

```powershell
uv run video-learner convert "C:\Users\cqqqwq\Videos\bilibili\1272065375" --start 00:14:30 --end 00:15:30 --profile math --provider qwen --asr-secret "C:\projects\video-learner\secrets\aliyun.secret" --secret "C:\projects\video-learner\secrets\aliyun.secret" --output "C:\projects\video-learner\output\my-qwen38-notes"
```

Qwen 图文请求使用 Chat Completions，默认 `enable_thinking=true`、`stream=true`。TOML 可设置 `qwen_enable_thinking` 和 `qwen_thinking_budget`（默认 1024）。终端显示思考、回答阶段；只收集最终正文，完成后校验 JSON、证据 ID 和资源引用。原始思考文本不写入日志或讲义。语音切片的来源精度不因更换图文模型而改变。

转换和文字修订均支持 `--model` 覆盖模型。显式切换 `--provider` 时会清除继承的旧模型、密钥环境变量名和密钥文件路径，再应用本次明确指定的值；同一 TOML 内显式填写的 `model`、`api_key_env` 仍需与供应商匹配。已有 DeepSeek 产物可以切换到 Qwen 修订：

```powershell
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r001 --block fig-001-002 --instruction "压缩图注为一句话" --provider qwen --secret "C:\projects\video-learner\secrets\aliyun.secret"
```

## 修订文字

从当前 Markdown 的章节标题和块标记取得 ID。默认基线为 r001，不会隐式选择最新版。

```powershell
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r001 --section ch-001 --instruction "保留关键步骤，压缩重复说明" --secret "C:\projects\video-learner\secrets\deepseek.secret"
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r002 --block blk-001-001 --instruction "修正术语并说明原课中的依据" --secret "C:\projects\video-learner\secrets\deepseek.secret"
```

目标的当前手改内容作为输入；非目标 Markdown 字节保留。章节修订保留标题，以避免改写范围外的目录。锚点缺失、重复、嵌套冲突或请求期间再次编辑会停止发布，诊断与建议片段在 `.work/conflicts/`。

可以手改 `notes.md`，不要改写 `notes.json` 和 `.work/`。手工增加的图片应保存在当前版本目录内，不能引用越界路径、外部 URL 或缺失文件。保留的手改块会标记结构化正文未核验同步。

## 精确换图与裁剪

```powershell
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r002 --block fig-001-002 --frame 00:46:12 --crop 640,0,640,410
```

图片块 ID 以实际输出为准。只提供 `--crop` 时复用原实际帧时间；只提供 `--frame` 时保留原裁剪。替换帧须在当前转换范围内，裁剪使用原帧像素。精确换图不读取模型凭据、不转写、不发 AI 请求。图注保留并加入待核对，受控图片行或来源行被改坏时报告冲突。

修订生成 `revisions/r002/` 等独立目录，包含本版 Markdown、图片、结构化索引、来源、疑点和 `changes.md`，旧文件保持不变。复制任一版本的 `notes.md` 和 `assets/` 即可阅读；自加图片在其他相对目录时一并复制该目录。

## 失败与检查

阶段日志走 stderr，JSON 和成功路径走 stdout。退出码：0 成功，1 运行/部分失败，2 输入或配置错误，130 用户中断。部分失败会列出未完成章节，不登记成功版本。P0 没有 `resume` 或 `status`；重新转换需指定新目录。

云端每次输入、输出和总调用次数均有限制；日志记录服务返回的 token 用量，不估算金额。DeepSeek 默认非思考模式，Qwen 默认开启思考；各自参数在 TOML 配置。超时调用可能仍计费。认证/参数错误和输出截断直接报告；临时网络错误及无效结构有界重试，流式连接失败后丢弃未完成正文。

开发验证：`uv run pytest`、`uv run ruff check .`、`uv run ruff format --check .`。默认测试使用自造媒体和确定性模型替身，不访问网络、GPU、凭据或私人课程。
