# 视频转 Markdown CLI：技术设计

状态：P0 已有运行实现，真实内容质量与整课验收状态见[实现指南](implementation-guide.md)。日期：2026-09-07。

本文只记录当前技术契约。需求边界以[用户故事](user-stories.md)为准；安装操作见[使用指南](usage.md)；过程与 ADR 见[实现笔记](implementation-notes.md)。

## 1. 架构与依赖

单视频/单课时缓存 → 图文 Markdown → 核对 → 局部修订 → 独立版本。P0 为 US-01/02/03/04/05/08/10，命令为 `inspect`、`convert`、`revise`。没有批量课程、前端、问答、学习管理、知识库或练习系统，US-09 保持删除。

| 职责 | 当前实现 |
| --- | --- |
| 环境 | Python 3.12、uv、uv.lock；Windows 为首个验证平台 |
| CLI | Typer/Rich，参数、错误提示与 stderr 阶段事件 |
| 应用 | workflows/conversion.py 编排转换，workflows/revision.py 编排修订；不依赖 Typer 对象 |
| 媒体 | PyAV，只读偏移流、轨道探测、seek、音频重采样 |
| 图片 | Pillow/NumPy，ROI、灰度变化、周期候选覆盖 |
| ASR | 阿里云 Qwen ASR，有界音频切片 |
| 多模态 | DeepSeek deepseek-v4-flash-vision-exp / Qwen qwen3.8-flash，窄 Provider.compose 接口 |
| 文档 | Pydantic、确定性 Markdown 渲染、字节范围替换、Markdown 资源解析 |
| 状态 | JSON、内容指纹、快照、临时目录、OS 文件锁 |
| 验证 | pytest 自造媒体/模型替身、Ruff、独立真实样本评估 |

代码按职责划分为 common（共享基础）、workflows（用例编排）、media（媒体与证据）、providers（云端模型）、notes（讲义组织与渲染）五个子包；根目录保留 cli.py。具体文件见[实现指南](implementation-guide.md#3-当前代码组织)。不建设空模块、通用插件框架、通用 Agent 引擎或后台服务。P1 才包含 SQLite、检查点、任务恢复、自主证据补查与复杂布局识别。

## 2. 导入与规范时间线

支持普通单视频和已验证 Bilibili 单课时缓存。文件名只提供线索，编码以实际轨道探测为准。读取已知标题和 BVID，缺失标题回退到文件名；不读取或记录签名下载 URL。发现独立 SRT/VTT，弹幕不当作字幕。

`OffsetReader` 提供只读、可 seek 的偏移视图。仅当确切九个 ASCII `0` 和后续大小合法的 ISO BMFF `ftyp` 首 box 同时成立时，才跳过九字节。不对所有 `.m4s` 删头，不读整课到内存，不在源目录生成规范化副本。

内部使用整数微秒，规范零点为视频轨道起始时刻；音频保留相对于同一个零点的偏移。轨道保留原始 start PTS、time base、起始时刻与声明时长，不能分别将音视频归零。

抽帧先 seek 到不晚于请求的关键帧，再向前解码到不早于请求的第一帧，登记实际 PTS 和时间；越过 `[start,end)` 则报告无可用帧。片段和重采样全部映射回原课时间。[PyAV seek 文档](https://pyav.org/docs/stable/api/container.html)

`inspect` 分别显示元数据下载完成、媒体打开、前中后抽查解码和全片验证。`--full` 顺序解码所有轨道，检查损坏帧及提前结束；稀疏抽查不能证明整片完整。

## 3. 固定证据提取

音频按重采样 PTS 放回 16 kHz 单声道缓冲，保留静音、时间空隙和原课偏移。语音识别统一使用 Qwen，不提供本地模型、下载、VAD 或 GPU 推理路径。

默认使用 `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`，模型默认为 `qwen3-asr-flash`。16 kHz PCM16 WAV 以 Base64 发送，默认每 30 秒一个独立切片，可设 5–60 秒；不重叠、不按字数插值时间。该接口没有句级时间戳，`TranscriptSegment.alignment=audio_window` 表示真实上传区间，正文来源及疑点明确标注精度。局部切片可能截断词句，须人工核对。原始波形、切片映射和响应保存在工作目录。凭据独立、最多 500 次 ASR 调用（含重试），认证/截断直接失败，瞬时错误有界重试；不自动回退其他云服务。[用户指定的 Qwen 接口](https://platform.qianwenai.com/docs/api-reference/speech-recognition/qwen-asr/openai)

当前提取指纹格式为 v2。修订旧目录时先按原格式校验指纹，再读取必要设置并复用已有证据；兼容层不加载本地模型、不迁移或改写原目录。新配置不接受旧的本地模型及模式选项。

原始转写不会被 LLM 改写覆盖。真实术语、分块边界和字幕同步仍需核对，不把合法时间戳视为识别准确的证明。SRT/VTT 检查格式、顺序、边界、有效文本和覆盖比例；必须显式选择，才替代 ASR。

默认每 10 秒扫描全帧或用户 ROI，用低分辨率灰度均差去重。同时保留每章起点及至少每 30 秒的周期候选，避免缓慢板书的小面积变化被过滤。每章最多向模型提供 12 张候选，超过预算时按时间均匀选择；模型再选择必要插图。

每张候选保留原帧、裁剪图、父证据、原像素坐标、请求时间、实际 PTS、time base、规范零点和文件哈希。裁剪不增加源信息。当前没有独立 OCR、自动布局估计、模糊字符恢复或自主补帧；清晰度、教学意义和内容对应需要模型选择与人工核对。

## 4. 多模态适配与证据约束

图文整理通过 `provider` 选择 DeepSeek 或 Qwen，不调用 OpenAI 服务或自动回退。`openai` 包作为两家服务的兼容客户端。默认 DeepSeek `deepseek-v4-flash-vision-exp`，固定连接 `https://api.deepseek.com`。[DeepSeek 图像接口](https://api-docs.deepseek.com/guides/vision/)

DeepSeek 使用 Responses API 图像输入和 JSON schema 响应。Qwen 使用 Chat Completions，固定连接 `https://dashscope.aliyuncs.com/compatible-mode/v1`，默认 `qwen3.8-flash`；图片以 Base64 `image_url` 发送，schema 同时写入请求格式和系统提示。两者均执行 Pydantic 及跨字段校验。text 块的 frame_id 必须为 null；figure 块必须引用本次提供的 frame ID 并将其包含在证据列表中。同章不重复导出同一截图。模型不能控制本地路径、HTML 锚点或来源时间戳。[DeepSeek Responses API](https://api-docs.deepseek.com/api/create-response/)

DeepSeek 默认 `reasoning_effort = "none"`，单次最多 6000 输出 token，任务最多 80 次实际请求、最多 2 次重试，均可在 TOML 调整。SDK 内置重试关闭。输出截断直接失败；临时网络错误及结构/引用错误有界重试；认证和请求配置错误直接报告。日志记录服务返回的输入/输出 token 和耗时，不估算金额；超时请求仍可能计费。[DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)

Qwen 默认 `enable_thinking=true`、`stream=true`，思考预算默认 1024 token。只拼接 `delta.content`；思考增量仅触发阶段提示，不保存文本。必须收到正常停止标记且正文非空才进入校验；连接中断丢弃部分正文，重试仍计入调用上限。处理仅含 usage 的尾块，结束或异常均关闭流。TOML 可通过 `qwen_enable_thinking` 和 `qwen_thinking_budget` 调整；DeepSeek 的 `reasoning_effort` 不发送给 Qwen。[Qwen 兼容接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)、[Qwen 图像理解](https://platform.qianwenai.com/docs/developer-guides/multimodal/vision)

章节默认 180 秒，调用前登记完整覆盖计划。固定证据包包含目标范围的可引用转写、候选图、用户要求和相邻只读文本；相邻文本不提供可引用 ID，避免跨章重复整理。逐章组织，失败章节保持未完成。P0 不运行模型自主工具循环。

编程保留目标、修改、错误、修复和运行结果；数学保留原课中存在的假设、符号、关键推导、结论和条件；混合内容按实际材料组织，不强行填模板。内容分为原课整理、显式要求的 AI 补充解释和待核对。模糊公式、代码和未交代条件进入疑点清单。

课程文字、字幕、代码和命令均为数据；供应商没有可执行工具，素材不能授权执行程序、读取任意路径或修改规则。请求和产物不含真实密钥或签名下载 URL。机械校验只保证结构、证据边界及资源关系；事实支持、公式正确和步骤覆盖属于独立内容评估。

## 5. 数据与独立版本

Pydantic 对象为 Source、Track、TranscriptSegment、FrameEvidence、Chapter、NoteBlock、ReviewItem、Notebook。根目录初稿为 r001，修订放入 revisions/r002/ 等，清单保存父版本和内容哈希。

```text
output/sample/
  notes.md
  review.md
  assets/
  source.json
  transcript.jsonl
  notes.json
  revisions/r002/
    notes.md
    review.md
    source.json
    notes.json
    assets/
    changes.md
  .work/
    manifest.json
    source-local.json
    versions/r001.generated.md
    frames/
    audio/
    logs/events.jsonl
    model-responses/
    conflicts/
```

可携带来源信息不含本机绝对路径，本机源位置与完整指纹只存在 `.work/`。每版 notes.md 与 assets/ 可独立复制；手加图片位于其他安全相对目录时也要一起携带。

章节和块使用独立起止 HTML 注释锚点。定位器忽略 fenced code 与缩进代码中的类似字符串，校验唯一性、闭合、嵌套和章节归属。修订默认显式基线 r001，可通过 `--base` 指定其他版本，绝不暗选最新版本。

目标的当前手改文本作为修订输入；非目标 Markdown 按原字节保留，写入前重新检查基线哈希。章节修订保留标题，避免改写范围外的目录。保留的手改块标为 manual_unverified。用户新增的安全本地图片会复制；外部 URL、越界、缺失资源和不支持的结构均报错。

精确换图/裁剪直接调用媒体层，不转写、不读取 AI 凭据。当前换图限定在基线转换时间范围内。实际帧、证据与来源行同步更新，图注保留并进入待核对；受控图片/来源行已被手改时输出冲突诊断和建议片段，不猜测位置。

## 6. 配置、提交与失败

输出路径解析为绝对路径，禁止与源目录重叠。所有资源解析后必须仍位于允许目录内，包含 Windows junction/符号链接边界；模型路径不作为写入目标。

同目录操作使用 OS 文件锁；Windows 为 msvcrt 字节锁。保留锁文件，由 OS 在进程结束时释放持有状态。转换在相邻临时目录生成、校验后 rename 发布；修订在工作目录临时区准备后发布为独立新目录。JSON/Markdown 使用临时文件、flush/fsync 和原子替换。

失败/取消保留诊断与已完成证据，不登记伪成功版本。P0 没有阶段恢复，未完成目录不能用 revise 复用；重新转换使用新目录。完成证据复用不称为断点恢复。索引、来源或提取配置不一致时停止复用。

配置优先级为 CLI > TOML > 默认值。图文凭据按供应商读取 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY，也可用 `--secret`，ASR 凭据读取 DASHSCOPE_API_KEY / `--asr-secret`，不持久化密钥或凭据文件路径。转换和文字修订支持 `--provider`、`--model`；跨供应商覆盖配置时先清除继承的模型、凭据环境变量名及凭据路径，再应用本次显式值和供应商默认值。模型选择不属于提取指纹，旧 DeepSeek 基线可直接用 Qwen 修订。日志走 stderr，JSON/成功路径走 stdout；Windows 重定向输出显式使用 UTF-8。退出码为 0 成功、1 运行/部分失败、2 输入/配置错误、130 中断。

## 7. 验证与 P1 边界

默认测试不联网、不下载权重、不需要 GPU 或私人视频。使用自造媒体与确定性模型替身验证前缀、时基/偏移、损坏文件、引用/缓存、路径、取消、版本独立性、手改和并发编辑。

真实 ASR、DeepSeek、Qwen、PPT/编程/数学片段与约 100 分钟整课单独记录。结构正确、内容正确、时间对齐、手改保护、人工使用效率分开评估；未经人工实测的准确率或修订耗时不填写数值。

P1 在用户授权后增加 status/resume、SQLite 检查点、配置依赖失效、AI 自主补查、自然语言重新选图和复杂布局识别，当前均未实施。


## 运行计时

事件日志按每次操作记录 run_id、UTC 时间和单调时钟相对秒数，保留原有 stage/status/seconds 字段。图文请求增加章节、尝试次数、打包字节数与耗时、失败耗时和响应 ID；Qwen 流计时区分打开流、首块、思考首块、正文首块及结束，只记录延迟和计数，不保存思考文本。流内通用 API 错误以受控 TaskError 结束并记录时长，不回显原始服务消息、不自动重试。开发工具及测量边界见[性能分析](performance.md)。
