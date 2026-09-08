import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier, Event, Lock

import pytest

from video_learner.common.concurrency import ordered_map
from video_learner.common.core import TaskError
from video_learner.common.storage import Events


def test_ordered_map_overlaps_without_unbounded_prefetch():
    """首批任务通过屏障证明重叠，消费暂停时不继续读取后续输入。"""
    gate = Barrier(3)
    seen = []

    def inputs():
        """记录预取量，发现一次性展开全部输入的内存回归。"""
        for index in range(9):
            seen.append(index)
            yield index

    def work(index):
        """每批三项同时进入后才返回，不依赖耗时阈值判断并发。"""
        gate.wait(timeout=3)
        return index * 2

    with closing(ordered_map(work, inputs(), 3, Event())) as results:
        assert next(results) == 0
        assert seen == [0, 1, 2]
        assert [0, *results] == list(range(0, 18, 2))


@pytest.mark.parametrize("consumer_error", [RuntimeError, KeyboardInterrupt])
def test_consumer_failure_waits_for_workers(consumer_error):
    """写入方异常或中断时，退出迭代器必须先取消并等候其余任务收尾。"""
    started, stopped, cancelled = Event(), Event(), Event()

    def work(index):
        """第二项模拟可取消的在途工作，退出前记录其 finally 已执行。"""
        if index == 0:
            assert started.wait(3)
            return index
        started.set()
        try:
            assert cancelled.wait(3)
        finally:
            stopped.set()
        return index

    with pytest.raises(consumer_error):
        with closing(ordered_map(work, range(10), 2, cancelled)) as results:
            assert next(results) == 0
            raise consumer_error()
    assert stopped.is_set()


def test_first_worker_failure_stops_requests_and_keeps_original_error():
    """后一任务失败时取消等待中的前项，主线程报告真实失败而非取消派生错误。"""
    cancelled = Event()
    gate = Barrier(2)
    seen = []
    lock = Lock()

    def work(index):
        """首项模拟重试等待，第二项模拟服务拒绝，剩余项不应被派发。"""
        with lock:
            seen.append(index)
        gate.wait(timeout=3)
        if index == 1:
            raise TaskError("原始服务失败")
        assert cancelled.wait(3)
        raise TaskError("已取消")

    with pytest.raises(TaskError, match="原始服务失败"):
        list(ordered_map(work, range(10), 2, cancelled))
    assert sorted(seen) == [0, 1]


def test_events_serialize_callbacks_and_keep_every_usage_record(tmp_path):
    """回调主动让出线程，证明日志和显示被同一锁串行保护，且没有丢失用量事件。"""
    active = False

    def observe(record):
        """重叠进入会直接失败，避免仅凭最终日志完整误判终端回调安全。"""
        nonlocal active
        assert not active
        active = True
        time.sleep(0.001)
        active = False

    events = Events(tmp_path, observe)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(events.emit, "asr_model_call", "running", call=i) for i in range(20)]
        for future in futures:
            future.result()
    saved = [json.loads(line) for line in events.path.read_text(encoding="utf-8").splitlines()]
    assert saved == events.usage_events
    assert sorted(e["call"] for e in saved) == list(range(20))
