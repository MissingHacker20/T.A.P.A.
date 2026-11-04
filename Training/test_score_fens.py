import io
from pathlib import Path
import unittest
from unittest.mock import patch

from score_fens import DIEngine


class FakeInput:
    def __init__(self) -> None:
        self.commands = []
        self.pending = ""
        self.closed = False

    def write(self, text: str) -> int:
        self.pending += text
        return len(text)

    def flush(self) -> None:
        self.commands.extend(line for line in self.pending.splitlines() if line)
        self.pending = ""

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, response: str) -> None:
        self.stdin = FakeInput()
        self.stdout = io.StringIO(response)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None) -> int:
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -1

    def kill(self) -> None:
        self.returncode = -9


class DIEngineTests(unittest.TestCase):
    def test_sends_startup_mode_and_returns_millipawn_score(self) -> None:
        process = FakeProcess("command entered\ncommand entered\n+9717 mP\n")
        with patch("score_fens.subprocess.Popen", return_value=process) as popen:
            with DIEngine(Path("MissingPawn.exe"), timeout=1) as engine:
                score, error = engine.evaluate_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")

        self.assertEqual(score, 9717)
        self.assertIsNone(error)
        self.assertEqual(popen.call_args.args[0], ["MissingPawn.exe", "di"])
        self.assertEqual(
            process.stdin.commands,
            [
                "fen 4k3/8/8/8/8/8/8/Q3K3 w - - 0 1",
                "eval 0 w",
            ],
        )

    def test_does_not_evaluate_after_fen_rejection(self) -> None:
        process = FakeProcess("error 03\n")
        with patch("score_fens.subprocess.Popen", return_value=process):
            with DIEngine(Path("MissingPawn.exe"), timeout=1) as engine:
                score, error = engine.evaluate_fen("invalid")

        self.assertIsNone(score)
        self.assertEqual(error, "error 03")
        self.assertEqual(process.stdin.commands, ["fen invalid"])


if __name__ == "__main__":
    unittest.main()
