# 使用指南

## 安装

项目要求 Python 3.12 和 uv，优先验证 Windows。在项目根目录运行 `uv sync --locked`，然后使用 `uv run video-learner`。普通素材检查不加载 ASR 模型或读取 AI 凭据。

## 配置

图文供应商通过 `--provider deepseek|qwen` 或 TOML 的 `provider` 选择。默认 DeepSeek，固定连接 `https://api.deepseek.com`，模型为 `deepseek-flash`；Qwen 固定连接 `https://dashscope.aliyuncs.com/compatible-mode/v1`，模型为 `qwen3.8-flash`。不自动切换服务，`openai` 包仅作为这两家服务的兼容客户端。

DeepSeek 凭据从 `DEEPSEEK_API_KEY` 读取，Qwen 从 `DASHSCOPE_API_KEY` 读取；也可使用 `--secret` 指定所选供应商的密钥文件（仅包含密钥的 UTF-8 文本）。不要将真实密钥写入 TOML、命令行参数或文档。仓库忽略 `secrets/`。

`--config` 指定 TOML；优先级为命令行 > TOML > 默认值。可参考 [config.example.toml](../config.example.toml)。明确要求额外解释时使用 `--instruction` 并加 `--allow-ai-additions`；生成内容会单独标为 AI 补充解释。

## 检查素材

直接运行 `uv run video-learner` 或添加 `--help` 可查看命令帮助，退出码为 0。查看子命令参数可运行 `uv run video-learner inspect --help`（`convert`、`revise` 同理）；未知命令、未知选项或缺少必填参数时退出码为 2，并显示参数错误。

```powershell
uv run video-learner inspect "C:\Users\cqqqwq\Videos\bilibili\41301577497" --json --decode
```

`--decode` 抽查前中后音视频，`--full` 顺序验证全部音视频解码。元数据下载完成、媒体打开、片段解码与全片验证分别显示。支持普通单视频文件及已知 Bilibili 单课时缓存，不递归处理多个课时，不下载在线视频。

## 转换

```powershell
uv run video-learner convert "C:\Users\cqqqwq\Videos\bilibili\41301577497" --start 00:45:00 --end 00:55:00 --profile programming --secret "C:\projects\video-learner\secrets\deepseek.secret" --output "C:\projects\video-learner\output\sample"
```

时间可用秒数或 `HH:MM:SS`，最多六位小数，按 `[start,end)` 处理。省略范围转换整课。输出目录必须与源目录不重叠。默认拒绝已存在的目录；加 `-f` / `--force` 会在预检通过并取得目录锁后删除整个原输出，再从头转换。旧版本、手改、日志均不保留，转换失败不会恢复旧结果。强制覆盖拒绝磁盘根目录、当前工作目录及其上级、符号链接或目录联接，也拒绝将本次字幕或凭据文件一并删除。

`--profile` 支持 `programming`、`math`、`mixed`；`--instruction` 设置组织要求。截图始终保留原分辨率完整画面。默认每 2 秒比较画面，明显变化时保留切换前一帧，并补周期帧和末帧，再由模型选图。短于采样间隔的画面可能遗漏，候选也允许冗余。`sample_seconds` 可调整间隔。

`--subtitle` 可指定 UTF-8 SRT/VTT 替代 ASR。字幕需时间合法、覆盖当前范围；覆盖比例写入工作记录，同步仍需人工核对。发现但未显式选择的字幕不会自动覆盖语音识别。

默认使用阿里云 Qwen ASR，按原视频时间提取 16 kHz 单声道音频并发送云端识别；不安装或下载本地识别模型。

生成的根目录对应 r001：`notes.md`、`assets/`、`sources.md`、`review.md`、`source.json`、`transcript.jsonl`、`notes.json`。正文保留连贯解释与必要图片，图注可省略；来源区间、证据和修订 ID 集中在 `sources.md`。`.work/` 保存本机来源、指纹、原帧、音频窗口、生成快照、版本清单与日志。

## 云端语音识别

转换默认使用 Qwen；ASR 与图文整理分别读取独立凭据：

```powershell
uv run video-learner convert "C:\Users\cqqqwq\Videos\bilibili\1272065375" --start 00:10:00 --end 00:20:00 --profile math --asr-secret "C:\projects\video-learner\secrets\aliyun.secret" --secret "C:\projects\video-learner\secrets\deepseek.secret" --output "C:\projects\video-learner\output\my-qwen-notes"
```

Qwen 凭据也可用 `DASHSCOPE_API_KEY`，与 DeepSeek 凭据独立。端点固定阿里云 DashScope，模型默认 `qwen3-asr-flash`。音频切片以 Base64 发送；默认每片上限 30 秒、最多 500 次调用，TOML 字段为 `asr_window_seconds`、`asr_max_calls`。窗口末尾 20% 内有至少 300 毫秒低音量停顿时，在停顿中间提前切分；没有则按长度切。静音不删除，尾片完整保留。提前切分可能增加少量调用。

`chapter_seconds` 默认 180 秒，是目标章长。章节在目标切点前后 20% 内尽量对齐转写结束位置，减少同一音频片段跨章重复。停顿与转写边界不等于话题边界。

该兼容接口不返回句级时间戳；来源明确标注为真实音频切片区间，切片边界可能截断词句。10 分钟数学音频实测转写约 32 秒，不是所有课程的固定性能。认证、调用上限和未完成输出会明确失败。识别设置进入提取指纹，修订时复用基线证据。工作目录的数据版本要求见下文精确换图说明。

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

从当前版本的 `sources.md` 取得章节和块 ID，也可查看 Markdown 源码中的隐藏锚点。默认基线为 r001，不会隐式选择最新版。

```powershell
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r001 --section ch-001 --instruction "保留关键步骤，压缩重复说明" --secret "C:\projects\video-learner\secrets\deepseek.secret"
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r002 --block blk-001-001 --instruction "修正术语并说明原课中的依据" --secret "C:\projects\video-learner\secrets\deepseek.secret"
```

目标的当前手改内容作为输入；非目标 Markdown 字节保留。章节修订保留标题，以避免改写范围外的目录。锚点缺失、重复、嵌套冲突或请求期间再次编辑会停止发布，诊断与建议片段在 `.work/conflicts/`。

可以手改 `notes.md`，不要改写 `notes.json` 和 `.work/`。手工增加的图片应保存在当前版本目录内，不能引用越界路径、外部 URL 或缺失文件。保留的手改块会标记结构化正文未核验同步。

## 精确换图

```powershell
uv run video-learner revise "C:\projects\video-learner\output\sample" --base r002 --block fig-001-002 --frame 00:46:12
```

图片块 ID 以实际输出为准。`--frame` 指定原视频时间，替换为该时刻起解码得到的完整画面。替换帧须在当前转换范围内。精确换图不读取模型凭据、不转写、不发 AI 请求。图注保留并加入待核对，来源索引同步更新；受控图片行已有手改时报告冲突。

修订只接受当前数据版本的工作目录，格式要求见[数据契约](technical-design.md#3-固定证据提取)。旧格式输出仍可阅读、复制，但不能直接继续修订；请重新转换到独立目录。

修订生成 `revisions/r002/` 等独立目录，包含本版 Markdown、图片、结构化索引、来源、疑点和 `changes.md`，旧文件保持不变。复制任一版本的 `notes.md` 和 `assets/` 即可阅读正文；需要保留来源与疑点链接时同时复制 `sources.md` 和 `review.md`。自加图片在其他相对目录时一并复制该目录。

## 失败与检查

阶段日志走 stderr，JSON 和成功路径走 stdout。退出码：0 成功，1 运行/部分失败，2 输入或配置错误，130 用户中断。部分失败会列出未完成章节，不登记成功版本。P0 没有 `resume` 或 `status`；重新转换需指定新目录，或显式加 `-f` 删除旧输出后重跑。

云端每次输入、输出和总调用次数均有限制。DeepSeek 默认非思考模式，Qwen 默认开启思考；各自参数在 TOML 配置。超时调用可能仍计费。认证/参数错误和输出截断直接报告；临时网络错误及无效结构有界重试，流式连接失败后丢弃未完成正文。

## Token meter 与估算费用

`convert` 结束时在终端显示各模型的请求次数、输入/输出 token、音频秒数和 Estimated 费用，并在输出目录保存 `usage.json`。成功、部分失败与已开始后的中断都会汇总已经记录的请求；预检失败尚未创建工作目录时不生成报告。费用和来源信息不插入讲义正文。

默认使用随包保存的[价格快照](../src/video_learner/common/prices.toml)，无需额外开关。该文件是当前单价、来源与日期的维护位置。图文按输入/输出 token 计费，缓存命中输入使用缓存价；ASR 按服务端返回的音频秒数计费。思考 token 已包含在服务返回的输出总量中，不再额外相加；图片同样使用服务计量，不自行按字符数换算。

DeepSeek 依据请求开始时的北京时间选择高峰/空闲价格；跨时段请求仍按开始时刻估算。缓存细分缺失时按普通输入价估算并提示，历史事件缺少开始时间时采用高峰价。缺失 token、计费秒数或单价的请求标为未估算；只算出部分费用时不显示为总价。报告包含所用单价，所有价格均为 estimated，以供应商账单为准，不自动联网更新价格。

修改单价可复制快照为自己的配置文件，再通过 `--config` 使用：

```powershell
Copy-Item src/video_learner/common/prices.toml config.prices.toml
# 编辑 config.prices.toml 中的单价，也可加入其他转换配置。
uv run video-learner convert "C:\path\lesson.mp4" --provider qwen --config config.prices.toml --output "C:\path\notes"
```

`prices` 的键是 `供应商:模型名`，例如 `qwen:qwen3.8-flash`。自定义表整体替换默认表，换到未报价模型时仍汇总 token，但不套用其他模型的单价。`input_per_million`、`output_per_million`、`cached_input_per_million` 按百万 token；`audio_per_second` 按秒，不能与 token 单价混用。`currency` 为报价币种，不同币种分别汇总，不转换汇率。未配置的字段表示未知，显式填零才表示免费。

当前报告对应本次 `convert`，不包含随后 `revise` 的费用；修订仍保留原有逐次请求用量日志。

开发验证：`uv run pytest`、`uv run ruff check .`、`uv run ruff format --check .`。默认测试使用自造媒体和确定性模型替身，不访问网络、GPU、凭据或私人课程。


## ASR 并发

`convert` 支持 `--jobs N`（1–32，默认 1），也可在 TOML 设置 `jobs`，命令行优先。例如在转换命令末尾加 `--jobs 4`。该参数仅控制 ASR 请求并发；图文整理继续按章节串行，使用外部字幕时不会启动 ASR 请求。

音频仍按顺序切分，结果按原切片编号保存；并发不改变切分、时间引用或图文上下文。所有 ASR 请求（包括重试）共享 `asr_max_calls` 上限。失败或 Ctrl+C 后停止派发和后续重试，等待正在进行的 HTTP 请求返回或超时再保存失败诊断，因此取消可能需要等待 `request_timeout_seconds` 所控制的网络超时。并发不代表任务恢复，也不会增加调用预算。

## 终端进度与详细日志

`convert` 和 `revise` 默认在交互终端原地刷新当前阶段、进度、总耗时及请求统计。语音识别按已完成音频时长推进；截图分别显示扫描与保存进度；图文整理显示成功章节数、失败数和当前模型活动。准备、校验与导出等没有可靠总量的阶段显示转圈，不估算整次任务的百分比。

请求统计显示已发出数、已结束数、累计重试和均耗时；重试累计显示，章节失败单独提示。结束后显示各阶段请求汇总、结果、总耗时及转换用量。加 `--verbose` 可打印详细阶段事件；输出重定向时默认仅打印阶段摘要，不输出动态控制字符。完整事件始终写入工作目录 `.work/logs/events.jsonl`，不需要额外开启日志；终端进度不通过轮询日志计算。


ASR 请求总量先按窗口上限估算，随后按已完成片段的平均时长动态更新；图文基础请求数按章节数计算。预计总请求包含已实际发出的重试，待发数量不含当前进行中的请求，也不预测未来重试。均耗时按有计时数据的已结束请求计算，包括失败请求；同一响应后续的校验或完成事件不重复计时。统计分别覆盖 ASR 与图文阶段，不把两者混为同一种请求。

阶段结束显示耗时：语音识别列出实际音频片数（含无文字切片）、请求数及重试数；截图列出扫描帧数与候选图数；图文列出成功章节数、插图数、请求数及重试数，部分失败单独标明。使用外部字幕时标注“外部字幕”，不显示音频片数。请求数包含重试，插图数按正文图片块统计。

音频策略、目标章长及停顿切点统计仅在 `--verbose` 中显示；默认仍显示实际划分章数。停顿切分默认开启，静音保留，不跳过全静音窗口。默认目标章长 180 秒，优先对齐附近转写结束边界；这不是语义话题识别。
