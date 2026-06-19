from __future__ import annotations

_WAIT_PREFIXES = ("<tool_call>", "<think>", "<answer>")


class AnswerStreamState:
    """
    Classifies LLM output chunks into: suppressed (tool call) or streaming (final answer).

    States:
      hold       — buffering chunks until the output type can be determined
      suppressed — <tool_call> detected; discard everything (caller handles full text)
      streaming  — final answer; emit chunks minus the tail guard
    """

    TAIL_GUARD = 12  # chars held back to prevent emitting closing </answer> tag

    def __init__(self) -> None:
        self._state: str = "hold"
        self._hold_buf: str = ""
        self._tail: str = ""

    def feed(self, chunk: str) -> str:
        """Feed one chunk; return text to emit as answer.delta (empty = nothing yet)."""
        if self._state == "suppressed":
            return ""
        if self._state == "streaming":
            return self._emit(chunk)
        # hold
        self._hold_buf += chunk
        return self._classify()

    def _classify(self) -> str:
        buf = self._hold_buf
        # Strip completed <think>…</think> blocks
        while True:
            think_start = buf.find("<think>")
            if think_start == -1:
                break
            think_end = buf.find("</think>", think_start)
            if think_end == -1:
                return ""  # think block not closed — keep buffering
            buf = buf[:think_start] + buf[think_end + len("</think>"):]

        stripped = buf.lstrip()
        if not stripped:
            return ""

        # Keep buffering if stripped could still be the start of a special opening tag
        for tag in _WAIT_PREFIXES:
            if tag.startswith(stripped) and stripped != tag:
                return ""

        if stripped.startswith("<tool_call>"):
            self._state = "suppressed"
            return ""

        # Final answer: start streaming
        self._state = "streaming"
        if stripped.startswith("<answer>"):
            stripped = stripped[len("<answer>"):]
        self._hold_buf = ""
        return self._emit(stripped)

    def _emit(self, chunk: str) -> str:
        combined = self._tail + chunk
        if len(combined) <= self.TAIL_GUARD:
            self._tail = combined
            return ""
        emit_text = combined[: -self.TAIL_GUARD]
        self._tail = combined[-self.TAIL_GUARD :]
        return emit_text

    def flush(self) -> str:
        """Drain the tail guard at stream end. Call exactly once after the last chunk."""
        if self._state == "hold":
            leftover = self._classify()
            tail, self._tail = self._tail, ""
            result = leftover + tail
        elif self._state == "suppressed":
            return ""
        else:
            result, self._tail = self._tail, ""

        # Strip trailing </answer> tag
        stripped = result.rstrip()
        if stripped.endswith("</answer>"):
            stripped = stripped[: -len("</answer>")].rstrip()
        return stripped

    @property
    def is_suppressed(self) -> bool:
        return self._state == "suppressed"
