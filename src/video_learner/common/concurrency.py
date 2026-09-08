"""有界并发执行，按输入顺序交付结果并在退出前收尾工作线程。"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from video_learner.common.core import TaskError


def ordered_map[T, R](
    action: Callable[[T], R], values: Iterable[T], jobs: int, cancelled: Event
) -> Iterator[R]:
    """最多保留 jobs 个任务，首个失败阻止后续派发；关闭迭代器时等待在途请求结束。

    输入迭代和结果消费均在调用线程进行；action 应在重试前检查 cancelled。
    不强杀线程，在途网络请求依靠调用方设置的超时结束。
    """
    failures: list[BaseException] = []
    failure_lock = Lock()

    def run(value: T) -> R:
        """保留首个实际失败，让其他任务取消时不会掩盖根因。"""
        if cancelled.is_set():
            raise TaskError("任务已取消")
        try:
            return action(value)
        except BaseException as exc:
            with failure_lock:
                if not failures:
                    failures.append(exc)
            cancelled.set()
            raise

    executor = ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="asr")
    pending = deque()
    inputs = iter(values)
    exhausted = False
    try:
        while not exhausted or pending:
            while not exhausted and len(pending) < jobs and not cancelled.is_set():
                try:
                    value = next(inputs)
                except StopIteration:
                    exhausted = True
                    break
                if not cancelled.is_set():
                    pending.append(executor.submit(run, value))
            if failures:
                raise failures[0]
            if cancelled.is_set():
                raise TaskError("任务已取消")
            if pending:
                try:
                    result = pending.popleft().result()
                except Exception:
                    if failures:
                        raise failures[0] from None
                    raise
                if failures:
                    raise failures[0]
                yield result
    finally:
        cancelled.set()
        executor.shutdown(wait=True, cancel_futures=True)
