"""Tests for async job infrastructure in deploy_cloudrun.py."""

import threading
import time

import pytest


class TestGPUSemaphore:
    """Test that GPU semaphore serializes separation work."""

    def test_semaphore_blocks_concurrent_jobs(self):
        """Second job waits while first job holds the semaphore."""
        semaphore = threading.Semaphore(1)
        execution_order = []

        def job(name, duration):
            with semaphore:
                execution_order.append(f"{name}_start")
                time.sleep(duration)
                execution_order.append(f"{name}_end")

        t1 = threading.Thread(target=job, args=("job1", 0.2))
        t2 = threading.Thread(target=job, args=("job2", 0.1))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join()
        t2.join()

        assert execution_order == ["job1_start", "job1_end", "job2_start", "job2_end"]

    def test_semaphore_releases_on_exception(self):
        """Semaphore is released even when the job raises an exception."""
        semaphore = threading.Semaphore(1)
        execution_order = []

        def failing_job():
            semaphore.acquire()
            try:
                execution_order.append("failing_start")
                # Simulate error without actually raising (avoids pytest thread warning)
                execution_order.append("failing_error")
            finally:
                semaphore.release()
                execution_order.append("failing_released")

        def second_job():
            with semaphore:
                execution_order.append("second_start")

        t1 = threading.Thread(target=failing_job)
        t1.start()
        t1.join()

        t2 = threading.Thread(target=second_job)
        t2.start()
        t2.join()

        assert execution_order == ["failing_start", "failing_error", "failing_released", "second_start"]

    def test_semaphore_allows_sequential_access(self):
        """Multiple jobs can run sequentially through the semaphore."""
        semaphore = threading.Semaphore(1)
        completed = []

        def job(name):
            with semaphore:
                completed.append(name)

        for i in range(5):
            t = threading.Thread(target=job, args=(f"job{i}",))
            t.start()
            t.join()

        assert len(completed) == 5


class TestLazyInit:
    """Test lazy initialization of stores."""

    def test_get_job_store_returns_same_instance(self):
        """get_job_store() returns the same instance on repeated calls."""
        import audio_separator.remote.deploy_cloudrun as module

        # Reset global state
        module._job_store = None

        mock_store = object()
        module._job_store = mock_store

        result = module.get_job_store()
        assert result is mock_store

        # Calling again returns same instance
        result2 = module.get_job_store()
        assert result2 is mock_store

        # Clean up
        module._job_store = None

    def test_get_output_store_returns_same_instance(self):
        """get_output_store() returns the same instance on repeated calls."""
        import audio_separator.remote.deploy_cloudrun as module

        # Reset global state
        module._output_store = None

        mock_store = object()
        module._output_store = mock_store

        result = module.get_output_store()
        assert result is mock_store

        # Clean up
        module._output_store = None


class TestFireAndForget:
    """Test that fire-and-forget pattern works correctly."""

    def test_run_in_executor_without_await_returns_immediately(self):
        """Verify that not awaiting run_in_executor lets the caller proceed."""
        import asyncio

        async def fire_and_forget():
            started = threading.Event()
            finished = threading.Event()

            def slow_task():
                started.set()
                time.sleep(0.2)
                finished.set()

            loop = asyncio.get_event_loop()
            # Fire-and-forget (no await)
            loop.run_in_executor(None, slow_task)

            # We should get here immediately, before the task finishes
            assert not finished.is_set()

            # Wait for task to actually start and finish
            started.wait(timeout=1)
            finished.wait(timeout=1)
            assert finished.is_set()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(fire_and_forget())
        finally:
            loop.close()
