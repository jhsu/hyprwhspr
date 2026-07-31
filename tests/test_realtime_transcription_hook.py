import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from realtime_transcription_hook import RealtimeTranscriptionHook


class NonClosingStringIO(io.StringIO):
    def close(self):
        self.flush()


class FakeProcess:
    def __init__(self):
        self.stdin = NonClosingStringIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class RealtimeTranscriptionHookTests(unittest.TestCase):
    def test_events_are_written_as_jsonl_and_hook_environment_is_limited(self):
        process = FakeProcess()
        with mock.patch(
            "realtime_transcription_hook.subprocess.Popen",
            return_value=process,
        ) as popen:
            hook = RealtimeTranscriptionHook(
                "hook-command",
                env={"HYPRWHSPR_MODEL": "gpt-live-transcribe"},
            )
            self.assertTrue(hook.start())
            self.assertTrue(hook.send({"event": "delta", "text": "hello"}))
            hook.stop()

        payload = json.loads(process.stdin.getvalue().strip())
        self.assertEqual(payload, {"event": "delta", "text": "hello"})
        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["HYPRWHSPR_MODEL"], "gpt-live-transcribe")
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])


if __name__ == "__main__":
    unittest.main()