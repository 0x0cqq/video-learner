# 视频转 Markdown CLI：技术设计

状态：P0 已有运行实现，实际验证与内容质量记录见[历史验证记录](validation.md)。

本文只记录当前技术契约。需求边界以[用户故事](user-stories.md)为准；安装操作见[使用指南](usage.md)；过程与 ADR 见[实现笔记](implementation-notes.md)。

## 1. 架构与依赖

单视频/单课时缓存 → 图文 Markdown → 核对 → 局部修订 → 独立版本。P0 为 US-01/02/03/04/05/08/10，命令为 `inspect`、`convert`、`revise`。没有批量课程、前端、问答、学习管理、知识库或练习系统，US-09 保持删除。

| 职责 | 当前实现 |
| --- | --- |
| 环境 | Python 3.12、uv、uv.lock；Windows 为首个验证平台 |
| CLI | Typer/Rich，参数、错误提示与 stderr 阶段进度；详细模式见[使用指南](usage.md#终端进度与详细日志) |
| 应用 | workflows/conversion.py 编排转换，workflows/revision.py 编排修订，workflows/replay.py 离线核对已完成结果；不依赖 Typer 对象 |
| 媒体 | PyAV，只读偏移流、轨道探测、seek、音频重采样 |
| 图片 | Pillow/NumPy，全帧灰度变化、周期候选覆盖 |
| ASR | 默认 Qwen，可选 faster-whisper CPU/CUDA，共用有界音频切片 |
| 多模态 | DeepSeek deepseek-flash / Qwen qwen3.8-flash，窄 Provider.compose 接口 |
| 文档 | Pydantic、证据包与草稿的确定性应用、Markdown 渲染、字节范围替换、资源解析 |
| 状态 | JSON、内容指纹、快照、临时目录、OS 文件锁 |
| 验证 | pytest 自造媒体/模型替身、Ruff、独立真实样本评估 |

代码按职责划分为 common（共享基础）、workflows（用例编排）、media（媒体与证据）、providers（模型适配）、notes（讲义组织与渲染）五个子包；根目录保留 cli.py。具体文件见[实现指南](implementation-guide.md#3-当前代码组织)。不建设空模块、通用插件框架、通用 Agent 引擎或后台服务。P1 才包含 SQLite、检查点、任务恢复、自主证据补查与复杂布局识别。

### 一次转换的数据流

`cli.convert_command()` 解析参数后调用 `workflows/conversion.py` 的 `convert()`。`convert()` 自行检查素材，不依赖用户先运行 `inspect`。预检完成后在输出目录旁创建临时工作目录；下表的路径均相对于该目录，成功提交后相对于用户指定的输出目录。完整目录结构见[数据与独立版本](#5-数据与独立版本)。

| 顺序 | 源码与数据变化 | 保存的结果 |
| --- | --- | --- |
| 1. 检查与准备 | `media/io.py:inspect_source()` 产生 `Source`；`conversion.py:fingerprint_source()` 核对实际媒体、元数据和显式字幕 | `.work/source-local.json` 保存本机来源与文件哈希；`.work/manifest.json` 保存运行状态、范围和配置；`.work/logs/events.jsonl` 追加阶段事件 |
| 2. 转写 | `providers/asr.py:transcribe()` 调用 `media/evidence.py:audio_window()`，按真实音频窗口取得 `TranscriptSegment`；显式 SRT/VTT 则由 `load_subtitles()` 提供片段 | `transcript.jsonl` 保存逐条转写证据；ASR 路径另存 `.work/audio/audio-*.wav`、同名 `.json` 时间映射及 `.work/asr-responses/audio-*.json` 识别文本；字幕路径另存 `.work/subtitle-check.json` |
| 3. 候选画面 | `media/evidence.py:sample_frames()` 扫描变化，`register_frame()` 登记实际 PTS、时间、图片路径和哈希，形成 `FrameEvidence` | `.work/frame-scan.json`、`.work/frames.json` 与 `.work/frames/frame-*.png`；这些是候选完整帧 |
| 4. 章节与模型前证据 | `notes/composition.py:plan_chapters()` 划分连续范围；`composition_input()` 从当前 `Notebook` 构造逐章证据包 | `.work/evidence.json` 保存整理前的转写、候选帧和章节计划；每章调用前写入 `.work/composition-inputs/ch-*.json`，包含实际证据包与图片身份 |
| 5. 模型与草稿应用 | `providers/base.py:Provider.compose()` 取得 `Draft`；`notes/composition.py:apply_chapter_draft()` 校验并应用草稿，分配章节内的稳定块 ID | `.work/model-responses/<随机ID>.json` 保存完整返回的最终正文，包含可能未通过校验的响应；`.work/composition-drafts/ch-*.json` 只保存已采用草稿；`.work/composed.json` 逐章更新讲义状态 |
| 6. 校验与发布 | `validate_notebook()` 检查时间、引用和图片；`notes/rendering.py:export_book()` 生成文稿与资源；`convert()` 复核源素材指纹后提交目录 | `notes.md`、`sources.md`、`review.md`、`notes.json`、`source.json`、`assets/`；`.work/versions/r001.generated.md` 保存生成字节快照，清单登记成功版本；`usage.json` 汇总本次用量 |
| 7. 独立复审 | `notes/reviewing.py:review_input()` 冻结基线和原证据；`Provider.review()` 返回逐图判断及局部修改；`apply_review_pass()` 确定性应用 | 默认写入独立 r002；`.work/reviews/` 保存复审基线、逐章输入和结果，可离线重放；原 r001 保留 |

图文模型逐章串行运行。后一章的证据包可包含已完成前章的末尾，因此每章必须在实际调用前单独冻结输入。章节在转写后规划一次，启动时报告目标章长。ASR 与图文模型输出是外部结果；给定保存的转写、候选帧和已采用 `Draft` 后，章节应用、渲染及固定结果的离线核对由本地代码完成。失败目录保留诊断，不登记成功版本，也不作为 P0 断点恢复入口。

## 2. 导入与规范时间线

支持普通单视频和已验证 Bilibili 单课时缓存。文件名只提供线索，编码以实际轨道探测为准。读取课程/课时标题和 BVID，有不同的 groupTitle 时组合为“课程 · 课时”，缺失标题回退到文件名；不读取或记录签名下载 URL。发现独立 SRT/VTT，弹幕不当作字幕。

`OffsetReader` 提供只读、可 seek 的偏移视图。仅当确切九个 ASCII `0` 和后续大小合法的 ISO BMFF `ftyp` 首 box 同时成立时，才跳过九字节。不对所有 `.m4s` 删头，不读整课到内存，不在源目录生成规范化副本。

内部使用整数微秒，规范零点为视频轨道起始时刻；音频保留相对于同一个零点的偏移。轨道保留原始 start PTS、time base、起始时刻与声明时长，不能分别将音视频归零。

抽帧先 seek 到不晚于请求的关键帧，再向前解码到不早于请求的第一帧，登记实际 PTS 和时间。自动采样在请求之后没有帧时，使用转换范围内的最后有效帧，分别保留原请求与实际时间，同一实际帧只登记一次；范围内没有可用帧或解码损坏时报告错误。精确换图仍要求帧时间位于 `[请求时间,end)`。片段和重采样全部映射回原课时间。[PyAV seek 文档](https://pyav.org/docs/stable/api/container.html)

`inspect` 分别显示元数据下载完成、媒体打开及轨道探测、前中后抽查解码和全片验证。`opened` 表示所有候选媒体均成功打开并完成轨道探测，不代表音视频轨道齐全或解码通过；部分媒体失败但仍有唯一有效视频轨道时返回诊断和 `opened=false`，CLI 退出码为 1。缺少唯一视频轨道或有效视频时长时直接报错。`--full` 顺序解码所有轨道，检查损坏帧及提前结束；稀疏抽查不能证明整片完整。

## 3. 固定证据提取

音频按重采样 PTS 放回 16 kHz 单声道缓冲，保留静音、时间空隙和原课偏移。语音识别默认 Qwen；显式 `asr_backend=local` 使用 faster-whisper。CPU 默认为 small/int8，CUDA 默认为 large-v3-turbo/int8_float16。两者通过 `recognize(wav, cancelled)` 与 `close()` 接口复用切片和保存流程，均按实际窗口引用。权重每次转写加载一次，按需下载；VAD 关闭，保留原始时间映射。CPU 的 `asr_cpu_threads` 控制算子线程，`jobs` 控制 CTranslate2 worker；CUDA 固定单 worker 控制显存。

默认使用 `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`，模型默认为 `qwen3-asr-flash`。16 kHz PCM16 WAV 以 Base64 发送，每片默认上限 30 秒，可设 5–60 秒。除尾片外，在窗口末尾 20% 内寻找至少 300 毫秒的低音量停顿（20 毫秒 RMS < 0.01），在最靠后的停顿中点切分；无停顿或全窗安静则保留完整窗口。下一片从实际切点开始，不丢静音、不重叠。固定阈值不等于语音检测，也不保证词句完整。该接口没有句级时间戳，`TranscriptSegment.alignment=audio_window` 表示真实上传区间，精度说明放在 `sources.md`。原始波形、切片映射和响应保存在工作目录。凭据独立、最多 500 次 ASR 调用（含重试），认证/截断直接失败，瞬时错误有界重试；不自动回退其他云服务。[用户指定的 Qwen 接口](https://platform.qianwenai.com/docs/api-reference/speech-recognition/qwen-asr/openai)

当前工作清单与讲义索引的结构版本为 2，提取版本为 4。每个截图证据只有一张完整帧缓存及其哈希。修订只接受当前数据版本并核对提取指纹，不加载或迁移旧结构；旧产物可独立阅读与复制，继续修订需重新转换到新目录。

ASR 通过 `jobs` 控制有界请求并发，默认 1；切片生成和响应保存仍按原时间顺序，空转写也保留原始响应。调用编号和含重试的总预算在锁内分配，事件日志、用量记录与显示回调串行写入。失败或取消后停止派发及后续重试，等待在途请求结束再关闭客户端、移动工作目录。`jobs` 和 CPU 线程数不影响提取指纹；本地模型、设备、beam 配置进入指纹，默认云端指纹保持稳定；图文整理仍按章节串行执行，上一章正文上下文保持原样。配置入口见[使用指南](usage.md#asr-并发)。

原始转写不会被 LLM 改写覆盖。真实术语、分块边界和字幕同步仍需核对，不把合法时间戳视为识别准确的证明。SRT/VTT 检查格式、顺序、边界、有效文本和覆盖比例；必须显式选择，才替代 ASR。

默认每 2 秒扫描完整画面，比较相邻 96×54 灰度缩略图的均差。变化达到 `image_change_threshold`（默认 0.035）时保留切换前一帧，同时保留起始帧、约每 30 秒的周期帧和最后一个采样帧。优先保留页面末态，但不推断“板书已经写完”；频繁切屏直接产生多个候选，不引入场景分类或复杂评分。短于采样间隔的画面仍可能遗漏。每章最多向模型提供 12 张候选，超过预算时均匀选择；模型再选择必要插图，允许候选冗余。

每张候选仅保存一份原分辨率完整帧，登记请求时间、实际 PTS、time base、规范零点和文件哈希。导出版本按实际引用复制图片，保证可独立携带。当前没有独立 OCR、自动布局估计、模糊字符恢复或自主补帧；清晰度、教学意义和内容对应需要模型选择与人工核对。

## 4. 多模态适配与证据约束

供应商返回的草稿由 `Draft` / `DraftBlock` 定义，响应 JSON schema 从该契约派生；正文通过结构和证据校验后再转成带稳定 ID 的讲义块。

图文整理通过 `provider` 选择 DeepSeek 或 Qwen，不调用 OpenAI 服务或自动回退。`openai` 包作为两家服务的兼容客户端。默认 DeepSeek `deepseek-flash`，固定连接 `https://api.deepseek.com`。[DeepSeek 图像接口](https://api-docs.deepseek.com/guides/vision/)

DeepSeek 使用 Responses API 图像输入和 JSON schema 响应。Qwen 使用 Chat Completions，固定连接 `https://dashscope.aliyuncs.com/compatible-mode/v1`，默认 `qwen3.8-flash`；图片以 Base64 `image_url` 发送，schema 同时写入请求格式和系统提示。两者均执行 Pydantic 及跨字段校验。text 块的 frame_id 必须为 null；figure 块必须引用本次提供的 frame ID 并将其包含在证据列表中。同章不重复导出同一截图。模型不能控制本地路径、HTML 锚点或来源时间戳。[DeepSeek Responses API](https://api-docs.deepseek.com/api/create-response/)

DeepSeek 默认 `reasoning_effort = "none"`，单次最多 6000 输出 token，任务最多 80 次实际请求、最多 2 次重试，均可在 TOML 调整。SDK 内置重试关闭。输出截断直接失败；临时网络错误及结构/引用错误有界重试；认证和请求配置错误直接报告。日志记录服务返回的输入/输出 token 和耗时；超时请求仍可能计费。[DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)

DeepSeek 转换默认逐轮追加已成功的原始请求（含图片）与回答，完整保留前缀；修订和 Qwen 保持独立请求。历史只用于术语和承接，引用校验仍仅允许当前包的证据。默认保守上下文预算 200K token、请求体预算 40 MiB，含输出及修复预留；到达边界整组重置，依靠包内上章末尾继续衔接。`deepseek_context=chapter` 可关闭历史。模型用量保留实际缓存 token；估计量只用于边界控制。

语义校验失败且带有历史时，修复轮清除历史并列出本章允许的证据 ID，计入原有调用预算。证据包附 is_last_chapter，提示词区分证据分组结束与课程内容缺失，要求核对数学条件、逻辑方向、评价归属及性能比较的负载口径；这些提示不替代内容核对。

Qwen 默认 `enable_thinking=true`、`stream=true`，思考预算默认 1024 token。只拼接 `delta.content`；思考增量仅触发阶段提示，不保存文本。必须收到正常停止标记且正文非空才进入校验；连接中断丢弃部分正文，重试仍计入调用上限。处理仅含 usage 的尾块，结束或异常均关闭流。TOML 可通过 `qwen_enable_thinking` 和 `qwen_thinking_budget` 调整；DeepSeek 的 `reasoning_effort` 不发送给 Qwen。[Qwen 兼容接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)、[Qwen 图像理解](https://platform.qianwenai.com/docs/developer-guides/multimodal/vision)

章节目标时长默认 180 秒，转写后在目标切点前后 20% 内选择最近的转写结束边界，再登记完整连续覆盖计划；没有合适边界则按长度切分，剩余不超过目标章长 20% 的尾段合入末章。它是证据分组，不代表语义话题检测。固定证据包包含本章转写、候选图、用户要求、相邻只读文本和上一章末尾至多 3000 字符；上下文不可引用，供承接与避免重复。逐章组织，失败章节保持未完成。P0 不运行模型自主工具循环。

编程保留目标、修改、错误、修复和运行结果；数学保留原课中存在的假设、符号、关键推导、结论和条件；混合内容按实际材料组织，不强行填模板。内容分为原课整理、显式要求的 AI 补充解释和待核对。模糊公式、代码和未交代条件进入疑点清单。

正文直接讲解课程内容，不逐图描述或堆放转写。图注允许为空，文字块必须非空；图意自明或已由正文解释时省略图注，不强求每章配图。提示词版本为 p0-7。原课类别、块 ID、来源区间、证据列表和手改同步状态放入 `sources.md`；`review.md` 列出内容疑点并以单条提示标明手改状态。正文仅显式标记 AI 补充和待核对，保留隐藏修订锚点。标题中的行内 LaTeX 保留原命令，标题其余 Markdown 控制字符转义。

证据包版本 2 用 `boundary_context_not_citable.before/after` 区分边界方向，每张帧的 `speech_window_ids` 指向其实际所在音频窗口。这个关系只表示时间邻近，不表示语义匹配，也不创造句级时间戳。正文确定性检查拒绝当前证据 ID 泄漏和章节末尾空标题，保留代码围栏中的字面示例；旧冻结包按原契约离线重建。

### 独立模型复审

`review_pass=true` 默认在 r001 提交后复审全部章节；`revise --review` 对显式成功基线复审整篇或指定章。复审逐章使用同一份冻结基线、实际 Markdown、原始转写、候选图及相邻章节首尾，不继承模型生成会话，也不让已修改章节改变后续复审输入。已有引用图片优先送入，剩余额度均匀补候选，图片预算与原提取配置一致。

`ReviewPass` 包含全部候选的 `FrameAssessment` 和局部 `ReviewFinding`。逐图判断给出采用/省略、支撑的原文字块、语音引用与理由；本地代码将采用的图片放在关联解释后，省略未选图，新增图默认无图注。文字修改、已有图注纠正和实质疑点分别记录为替换、插入或报告；证据和目标必须属于本次范围，同一块最多一次修改。基于稳定块 ID 比较新旧序列，仅拼接变化范围，保留其他字节；移动原图时保留手改图注。选图理由留在工作记录与 changes.md，未解决的疑点进入 review.md。模型复审仍需人工抽查事实和读图。

复审沿用目录锁、源指纹、并发编辑检查、版本独立导出和原子提交。初稿与复审共用 `max_calls` 预算，关闭复审只产生 r001。复审失败时保留成功初稿并报告失败，失败复审不登记新版本。输入、结果及基线存于 `.work/reviews/<记录ID>/`，清单关联 `review_record`；`replay_revision()` 重建逐章输入、图片身份及最终讲义。单独复审的用量按运行保存并复制到成功版本，转换中的复审计入根目录总用量。

课程文字、字幕、代码和命令均为数据；供应商没有可执行工具，素材不能授权执行程序、读取任意路径或修改规则。请求和产物不含真实密钥或签名下载 URL。机械校验只保证结构、证据边界及资源关系；事实支持、公式正确和步骤覆盖属于独立内容评估。

正文一级/二级标题在本地归为三级，使用 Markdown 解析定位以保留围栏代码、引用和 Setext 内容；原始模型响应保留不改，实际提交的上下文仍使用原始回答。

### 可复现的业务边界

转换在模型整理前保存含转写、候选帧和章节计划的 `.work/evidence.json`。每章实际使用的证据包及图片相对路径、SHA-256 组成结构版本 1 的 `CompositionInput`，在请求前写入 `.work/composition-inputs/`；通过校验并采用的 `Draft` 写入 `.work/composition-drafts/`。上一章的模型结果会影响下一章证据包，因此逐章按原顺序冻结。文字修订另保存调用时的结构化讲义和包含手改内容的 Markdown 基线，并在版本清单关联相应输入与草稿。原始模型最终正文仍保存在 `model-responses/`。

`notes/composition.py` 负责从证据形成输入，以及将给定草稿确定性地应用到转换章节或修订目标；`providers` 保持外部模型适配与有界调用职责。开发工具 `tools/replay_conversion.py` 只读重建成功转换的 r001 或指定文字修订：转换逐章核对冻结输入，修订核对目标手改文本与图片，均核对采用的草稿、结构化讲义与生成的 Markdown 快照。精确换图不调用模型，由现有媒体与修订测试验证。离线重建不重发 API 请求，也不代表模型内容正确；服务端模型、思考过程、DeepSeek 完整历史及修复轮的再次生成不保证相同。

### 独立人工校订

开发工具 export_review.py 接收显式人工校订的 Notebook，检查原始媒体、转写、图片及章节范围未变，逐章检查引用边界，再导出独立阅读副本。检测到基线 Markdown 手改时停止，输出目录必须全新；不修改基线、不登记成功版本，也不提供任务恢复。

## 5. 数据与独立版本

持久化数据的 Pydantic 对象为 Source、Track、TranscriptSegment、FrameEvidence、CompositionInput、Draft、ReviewPass、FrameAssessment、ReviewFinding、Chapter、NoteBlock、ReviewItem、Notebook。根目录初稿为 r001，修订放入 revisions/r002/ 等，清单保存父版本和内容哈希。

```text
output/sample/
  notes.md
  sources.md
  review.md
  assets/
  source.json
  transcript.jsonl
  notes.json
  usage.json
  revisions/r002/
    notes.md
    sources.md
    review.md
    source.json
    notes.json
    assets/
    changes.md
  .work/
    manifest.json
    source-local.json
    evidence.json
    composed.json
    composition-inputs/
    composition-drafts/
    revision-inputs/
    reviews/
    review-usage/
    versions/r001.generated.md
    frame-scan.json
    frames.json
    frames/
    audio/
    asr-responses/
    subtitle-check.json
    logs/events.jsonl
    model-responses/
    conflicts/
```

`audio/` 与 `asr-responses/` 仅在运行 ASR 时生成，`subtitle-check.json` 仅在显式使用字幕时生成；修订与冲突目录也按实际操作出现。可携带来源信息不含本机绝对路径，本机源位置与完整指纹只存在 `.work/`。每版 notes.md 与 assets/ 可独立复制；手加图片位于其他安全相对目录时也要一起携带。

章节和块使用独立起止 HTML 注释锚点。正文校验与定位器共用 Markdown 解析规则，支持列表、引用中的代码围栏；模型围栏必须显式闭合。定位器忽略 fenced code 与缩进代码中的类似字符串，校验唯一性、闭合、嵌套和章节归属。修订默认显式基线 r001，可通过 `--base` 指定其他版本，绝不暗选最新版本。

修订检查媒体/元数据指纹与轨道结构，展示标题的组合方式变化不使原证据失效。目标的当前手改文本作为修订输入；非目标 Markdown 按原字节保留，写入前重新检查基线哈希。章节修订保留标题，避免改写范围外的目录。保留的手改块标为 manual_unverified。用户新增的安全本地图片会复制；外部 URL、越界、缺失资源和不支持的结构均报错。

待核对项的 `block_id` 可关联章节或块，导出时检查目标存在。文字修订重建目标范围的疑点，包括 `uncertain` 正文与模型返回的提示；范围外条目和没有目标关联的全局提示保留。整章修订同时替换该章及原有块的疑点，精确换图保留已有疑点并追加图文一致性提示。

精确换图直接调用媒体层，不转写、不读取 AI 凭据。当前换图限定在基线转换时间范围内。实际帧、证据与独立来源索引同步更新，图注保留并进入待核对；受控图片行已被手改时输出冲突诊断。只替换目标图片行，图注、附加文字和其他块的字节保持不变。

## 6. 配置、提交与失败

输出路径解析为绝对路径，禁止与源目录重叠。所有资源解析后必须仍位于允许目录内，包含 Windows junction/符号链接边界；模型路径不作为写入目标。

同目录操作使用 OS 文件锁；Windows 为 msvcrt 字节锁。保留锁文件，由 OS 在进程结束时释放持有状态。转换在相邻临时目录生成、校验后 rename 发布；修订在工作目录临时区准备后发布为独立新目录。转换与修订共用一次导出流程来定位锚点、准备图片和校验依赖；生成快照使用实际导出的 Markdown 字节。JSON/Markdown 使用临时文件、flush/fsync 和原子替换。

失败/取消保留诊断与已完成证据，不登记伪成功版本。P0 没有阶段恢复，未完成目录不能用 revise 复用；重新转换默认使用新目录；显式 `convert -f / --force` 在输入及凭据预检通过、持有目录锁后删除原输出，再生成新结果，失败不恢复旧输出。删除前重新解析目标并核对路径边界；拒绝源目录重叠、磁盘根目录、当前工作目录及其上级、链接目录，以及包含本次字幕、凭据或本地模型的输出目录。完成证据复用不称为断点恢复。索引、来源或提取配置不一致时停止复用。

配置优先级为 CLI > TOML > 修订基线（修订时）> 默认值；读取基线保留合法空值，未提供的 CLI 参数不覆盖基线。图文凭据按供应商读取 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY，也可用 `--secret`，ASR 凭据读取 DASHSCOPE_API_KEY / `--asr-secret`，不持久化密钥或凭据文件路径。转换和文字修订支持 `--provider`、`--model`；跨供应商覆盖配置时先清除继承的模型、凭据环境变量名及凭据路径，再应用本次显式值和供应商默认值。模型选择不属于提取指纹，旧 DeepSeek 基线可直接用 Qwen 修订。日志走 stderr，JSON/成功路径走 stdout；Windows 重定向输出显式使用 UTF-8。退出码为 0 成功、1 运行/部分失败、2 输入/配置错误、130 中断。

## 7. 运行计时

`Events` 在当前运行内保留请求开始及返回用量事件，`common/usage.py` 按模型汇总转换用量和估价，CLI 负责表格显示。重试各计一次，输入/输出子项不重复相加；汇总不轮询日志，也不混入后续修订。转换退出时将 `usage.json` 写入最终或失败目录，包含用量缺口、已知部分费用与价格快照。计价配置不影响证据提取指纹。单价及字段只维护在[价格快照](../src/video_learner/common/prices.toml)和[使用说明](usage.md#token-meter-与估算费用)中。

事件日志按每次操作记录 run_id、UTC 时间和单调时钟相对秒数，保留原有 stage/status/seconds 字段。图文请求增加章节、尝试次数、打包字节数与耗时、失败耗时和响应 ID；Qwen 流计时区分打开流、首块、思考首块、正文首块及结束，只记录延迟和计数，不保存思考文本。流内通用 API 错误以受控 TaskError 结束并记录时长，不回显原始服务消息、不自动重试。开发工具及测量边界见[性能分析](performance.md)。

终端显示由 `terminal.py` 消费 Events 的结构化回调，业务层不依赖 Rich。进度事件携带 `completed` / `total`，音频使用本次范围的整数微秒，截图与章节使用计数；失败章节不计入成功数。日志与用量收集继续使用原事件，新增进度和重试事件不计入请求用量。

终端按调用编号合并请求计时，响应与校验事件不重复计入均耗时；ASR 失败事件也记录请求耗时。预计请求总量使用音频片段平均时长或实际章节数，累计已发出的重试；显示口径见[使用指南](usage.md#终端进度与详细日志)。启动策略和最终章数通过结构化事件报告。

## 8. 验证与 P1 边界

默认测试不联网、不下载权重、不需要 GPU 或私人视频。使用自造媒体与确定性模型替身验证前缀、时基/偏移、损坏文件、引用/缓存、路径、取消、版本独立性、手改和并发编辑。

真实 ASR、DeepSeek、Qwen、PPT/编程/数学片段与约 100 分钟整课单独记录。结构正确、内容正确、时间对齐、手改保护、人工使用效率分开评估；未经人工实测的准确率或修订耗时不填写数值。

P1 在用户授权后增加 status/resume、SQLite 检查点、配置依赖失效、AI 自主补查、自然语言重新选图和复杂布局识别，当前均未实施。
