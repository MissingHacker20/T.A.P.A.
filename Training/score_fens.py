#!/usr/bin/env python3
"""Batch-evaluate one FEN per line with MissingPawn's `di` interface."""

import argparse
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading


ACK = "command entered"
ERROR_RE = re.compile(r"^error\b", re.IGNORECASE)
SCORE_RE = re.compile(r"^([+-]?\d+)\s+mP$", re.IGNORECASE)


class EngineError(RuntimeError):
    pass


class DIEngine:
    def __init__(self, executable: Path, timeout: float) -> None:
        self.timeout = timeout
        self.process = subprocess.Popen(
            [str(executable), "di"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _read_line(self) -> str:
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise EngineError(f"timed out after {self.timeout:g}s waiting for engine response") from exc
        if line is None:
            raise EngineError(f"engine exited unexpectedly (code {self.process.poll()})")
        return line

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise EngineError(f"engine exited unexpectedly (code {self.process.returncode})")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError("could not send a command to the engine") from exc

    def _wait_for_ack(self) -> str:
        while True:
            line = self._read_line()
            if line == ACK or ERROR_RE.match(line):
                return line

    def evaluate_fen(self, fen: str) -> tuple[int | None, str | None]:
        self._send("fen " + fen)
        response = self._wait_for_ack()
        if ERROR_RE.match(response):
            return None, response

        self._send("eval 0 w")
        response = self._wait_for_ack()
        if ERROR_RE.match(response):
            return None, response

        while True:
            line = self._read_line()
            if ERROR_RE.match(line):
                return None, line
            match = SCORE_RE.fullmatch(line)
            if match:
                return int(match.group(1)), None

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join()
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self) -> "DIEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a positive number") from exc
    if timeout <= 0 or not timeout < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return timeout


def read_fens(path: Path) -> list[tuple[int, str]]:
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            return [
                (line_number, stripped)
                for line_number, line in enumerate(source, start=1)
                if (stripped := line.strip()) and not stripped.startswith("#")
            ]
    except (OSError, UnicodeError) as exc:
        raise EngineError(f"cannot read input file: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one FEN per input line through MissingPawn's di interface."
    )
    parser.add_argument("input", type=Path, help="text file containing one FEN per line")
    parser.add_argument(
        "-e",
        "--engine",
        type=Path,
        default=Path(__file__).resolve().with_name("MissingPawn.exe"),
        help="path to MissingPawn.exe (default: next to this script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output TSV path; by default, results are written to stdout",
    )
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=10.0,
        help="seconds to wait for each engine response (default: 10)",
    )
    args = parser.parse_args()

    try:
        fens = read_fens(args.input)
        if not fens:
            raise EngineError("input file contains no FEN positions")
        if not args.engine.is_file():
            raise EngineError(f"engine executable not found: {args.engine}")
        if args.output is not None and args.input.resolve() == args.output.resolve():
            raise EngineError("input and output paths must be different")

        output = args.output.open("w", encoding="utf-8", newline="\n") if args.output else sys.stdout
        close_output = args.output is not None
        succeeded = 0
        failed = 0
        try:
            with DIEngine(args.engine, args.timeout) as engine:
                for line_number, fen in fens:
                    score, error = engine.evaluate_fen(fen)
                    if error is not None:
                        print(f"line {line_number}: engine rejected FEN: {error}", file=sys.stderr)
                        failed += 1
                        continue
                    assert score is not None
                    output.write(f"{fen}\t{score}\n")
                    output.flush()
                    print(f"line {line_number}: OK, {score:+d} mP", file=sys.stderr)
                    succeeded += 1
        finally:
            if close_output:
                output.close()

        print(f"Evaluated {succeeded} FEN(s); {failed} error(s).", file=sys.stderr)
        return 1 if failed else 0
    except (EngineError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
