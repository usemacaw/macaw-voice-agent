"""Thread pool executor for async/sync inference boundary.

Limits concurrent inference threads to prevent CPU/GPU contention.
GPU serializes operations — more threads don't help.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


def create_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    """Create a bounded thread pool for inference work."""
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="macaw-asr-inference",
    )


async def run_in_executor(
    executor: ThreadPoolExecutor, fn, *args, **kwargs
):
    """Run a blocking inference function in the thread pool.

    Args:
        executor: Thread pool to run in.
        fn: Blocking function (e.g. model.generate).
        *args, **kwargs: Arguments for fn.

    Returns:
        Result of fn(*args, **kwargs).
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(
            executor, lambda: fn(*args, **kwargs)
        )
    return await loop.run_in_executor(executor, fn, *args)
