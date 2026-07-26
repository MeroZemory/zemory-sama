from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from zemory import __main__ as cli
from zemory.config import RuntimeCredentialError


def test_cli_reports_missing_credentials_without_traceback(monkeypatch, capsys) -> None:
    async def fail() -> None:
        raise RuntimeCredentialError("OPENAI_API_KEY is required")

    monkeypatch.setattr(cli, "run", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out == ""
    assert captured.err == "Configuration error: OPENAI_API_KEY is required\n"
    assert "Traceback" not in captured.err


def test_cli_runner_exits_when_provider_suppresses_cancellation() -> None:
    script = textwrap.dedent(
        """
        import asyncio

        from zemory import __main__ as cli
        from zemory.pipeline import tts_manager as manager_module
        from zemory.pipeline.tts_manager import TTSTaskManager

        manager_module._TASK_SHUTDOWN_TIMEOUT_S = 0.01

        class Queue:
            async def put(self, payload: bytes) -> None:
                return None

        class Speaker:
            def __init__(self) -> None:
                self.queue = Queue()

        class CancellationResistantTTS:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def synthesize(self, text: str, quick: bool = False):
                self.started.set()
                while True:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        continue
                if False:
                    yield b""

        async def workload() -> None:
            tts = CancellationResistantTTS()
            manager = TTSTaskManager(tts, Speaker(), max_concurrent=1)
            manager.start()
            manager.submit("private text")
            await tts.started.wait()
            await manager.stop()

        cli._run_with_bounded_shutdown(workload(), shutdown_timeout_s=0.05)
        print("runner-returned")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=2.0,
    )

    assert completed.returncode == 0
    assert completed.stdout.endswith("runner-returned\n")
    assert "private text" not in completed.stdout
    assert "private text" not in completed.stderr
    assert "abandoned 1 cancellation-resistant" in completed.stderr


def test_cli_preserves_keyboard_interrupt_exit(monkeypatch, capsys) -> None:
    async def interrupt() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", interrupt)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.out == "\nGoodbye!\n"
