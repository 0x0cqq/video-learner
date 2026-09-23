# 性能分析工具

`tools/profile_conversion.py` 从完成任务的事件日志汇总阶段耗时、请求延迟和失败类别，不调用模型。使用标准库 cProfile、pstats，可离线回放本地阶段，所有输出写入独立报告目录。

```powershell
uv run python tools/profile_conversion.py --help
```

默认只分析日志。加 `--local` 后按原课中间的一段回放音频准备、截图采样，并对讲义回放校验、渲染和导出；该模式要求当前数据版本的成功转换目录。素材与已有讲义保持只读。

ASR 的并发设置见[使用指南](usage.md#asr-并发)，图文整理保持串行。截图每个证据只保存一张完整帧；选帧策略与实际帧时间映射保持明确。

测量时区分墙钟耗时与进程 CPU 时间，避免重复叠加嵌套阶段。报告中的请求累计耗时可以包含并发重叠，不用它除以阶段墙钟时间计算占比。样本范围、缓存控制及运行次数应随结果记录；本地测量不能用于推断云端并发性能。已执行验证见[历史验证记录](validation.md)，早期整课分析与双文件缓存测量集中保存在[实现笔记](implementation-notes.md)。

## 本地 ASR 短片段测量

使用 `uv run --extra local-asr python tools/profile_asr.py <单课时目录> --start 300 --seconds 60 --threads 4 --jobs 2 --output artifacts/asr/my-run`。CUDA 添加 `--device cuda --jobs 1` 并按使用指南设置运行库。工具只识别指定 5–180 秒片段，不调用图文或云端 ASR；记录模型加载、冷/热运行、wall/CPU 耗时、实时率（耗时/音频时长）、进程 RSS 和 100 ms 采样的整卡显存。首次下载与模型加载须分别看待；RSS/显存峰值覆盖加载和推理，WDDM 整卡数据包含其他程序。

已执行的 CPU/CUDA 测量与适用范围见[历史验证记录](validation.md#2026-09-14-短片段性能测量)。

## 冻结证据回放

`uv run python tools/replay_composition.py output/sample --output artifacts/replay/example --sections ch-001 ch-002` 默认只准备证据包，不创建模型客户端。新产物使用原始调用前冻结的章节输入；较早产物从现有讲义索引重组并提示。加 `--live --secret secrets/deepseek.secret` 才发送指定章节；每次最多三章、无重试。`--context chapter|history` 比较独立逐章与所选章节历史策略，不重建未选章节或原始修复轮的完整历史。新输出保存 packet、draft 和 usage，输入讲义及其证据保持只读，不重复 ASR 或媒体采样。

固定模型结果的确定性回归使用 `uv run --no-sync python tools/replay_conversion.py output/sample`；文字修订加 `--revision r002`。它重建结构化讲义并核对生成快照，不进行性能计时或模型质量评价。

真实小范围对照数据见[验证记录](validation.md#固定证据的缓存对照)。缓存节省应同时观察未命中输入、已命中输入、输出长度及总费用；高命中率本身不是净省费证明。

同机四课 CUDA 整段识别与阶段计时见[历史整课验证](validation.md#四个完整课时与学习者审阅)。
