"""Bounded JSONL observer hook for realtime transcription events."""

import os
import json
import subprocess
import threading
from queue import Full, Queue
from typing import Dict, Optional


class RealtimeTranscriptionHook:
    """Write realtime events to one user command without blocking audio threads."""

    _STOP = object()
    _QUEUE_SIZE = 128
    _STOP_TIMEOUT = 5.0

    def __init__(self, command: Optional[str], env: Optional[Dict[str, str]] = None):
        self.command = command.strip() if isinstance(command, str) else ''
        self.env = dict(env or {})
        self._lock = threading.Lock()
        self._process = None
        self._queue = None
        self._writer_thread = None
        self._accepting = False

    def start(self) -> bool:
        """Start the hook if configured; return whether it is available."""
        if not self.command:
            return False

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return True

        # A crashed hook may leave its writer thread behind. Reap that
        # generation before replacing the queue/process references.
        self.stop()

        with self._lock:
            self._queue = Queue(maxsize=self._QUEUE_SIZE)
            self._accepting = True

            # Pass only the basic execution environment plus documented hook
            # metadata; do not leak API keys or unrelated service variables.
            hook_env = {
                key: os.environ[key]
                for key in ('HOME', 'LANG', 'PATH', 'SHELL', 'USER')
                if key in os.environ
            }
            hook_env.update(self.env)
            try:
                self._process = subprocess.Popen(
                    self.command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=None,
                    text=True,
                    env=hook_env,
                    start_new_session=True,
                )
            except Exception as exc:
                self._accepting = False
                self._queue = None
                print(f'[REALTIME] Failed to start transcription hook: {exc}', flush=True)
                return False

            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
                name='RealtimeTranscriptionHook',
            )
            self._writer_thread.start()
            return True

    def send(self, event: dict) -> bool:
        """Queue one event without blocking the WebSocket receiver thread."""
        with self._lock:
            if not self._accepting or self._queue is None:
                return False
            queue = self._queue

        try:
            queue.put_nowait(event)
            return True
        except Full:
            # Dropping an intermediate delta is preferable to stalling audio;
            # terminal/boundary events make one best-effort retry after evicting
            # an older queued event.
            if event.get('event') == 'delta':
                return False
            try:
                queue.get_nowait()
                queue.task_done()
                queue.put_nowait(event)
                return True
            except Exception:
                return False

    def stop(self) -> None:
        """Stop the hook process and discard no already-queued events."""
        with self._lock:
            if self._process is None and self._queue is None:
                return
            self._accepting = False
            queue = self._queue
            process = self._process
            writer = self._writer_thread

        try:
            queue.put(self._STOP, timeout=self._STOP_TIMEOUT)
        except Exception:
            pass
        if writer and writer is not threading.current_thread():
            writer.join(timeout=self._STOP_TIMEOUT)

        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=self._STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                    process.wait(timeout=1.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass

        with self._lock:
            self._process = None
            self._queue = None
            self._writer_thread = None

    def _writer_loop(self):
        while True:
            with self._lock:
                queue = self._queue
                process = self._process
            if queue is None or process is None:
                return

            item = queue.get()
            try:
                if item is self._STOP:
                    return
                if process.poll() is not None or process.stdin is None:
                    return
                process.stdin.write(json.dumps(item, ensure_ascii=False) + '\n')
                process.stdin.flush()
            except Exception as exc:
                print(f'[REALTIME] Transcription hook stopped: {exc}', flush=True)
                with self._lock:
                    self._accepting = False
                return
            finally:
                queue.task_done()