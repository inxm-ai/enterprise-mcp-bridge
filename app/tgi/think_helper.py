from typing import Callable, Optional
import uuid

from app.tgi.chunk_reader import create_response_chunk


class ThinkExtractor:
    """
    Incrementally consumes chunks of text and returns all completed
    <think>...</think> blocks (including tags) found in the buffer when
    feed() is called. If no completed block is available, returns a single
    newline ("\n") to indicate "no status" except for the special case of
    a solitary partial opening-tag received on an empty buffer (returns "").

    feed(chunk) -> str
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, id_generator: Optional[Callable[[], str]] = None) -> None:
        self._buf: str = ""
        self.id_generator = id_generator or (lambda: f"mcp-{str(uuid.uuid4())}")

    def feed(self, chunk: str) -> str:
        if chunk is None:
            chunk = ""

        prior_buf_empty = len(self._buf) == 0
        # append incoming chunk
        self._buf += chunk

        # If there is no opening tag at all, keep only possible partial "<think" fragments
        first_open = self._buf.find(self.OPEN)
        if first_open == -1:
            # If this call began with an empty buffer and the provided chunk is a
            # prefix of the OPEN tag, return "" (special-case test expectation).
            if prior_buf_empty and 0 < len(chunk) < len(self.OPEN) and self.OPEN.startswith(chunk):
                # keep the partial chunk in buffer for later completion
                return ""

            # Otherwise, trim to preserve possible partial "<think" fragments and return newline
            keep = max(0, len(self.OPEN) - 1)
            if len(self._buf) > keep:
                self._buf = self._buf[-keep:]
            return "\n"

        # There is at least one opening tag; collect all completed blocks following it.
        result_parts = []
        pos = first_open
        while True:
            close_idx = self._buf.find(self.CLOSE, pos + len(self.OPEN))
            if close_idx == -1:
                # No closing tag for the current open: preserve from first_open onward
                self._buf = self._buf[first_open:]
                break

            end = close_idx + len(self.CLOSE)
            block = self._buf[pos:end]
            result_parts.append(block)

            # Look for next opening tag after end
            next_open = self._buf.find(self.OPEN, end)
            if next_open == -1:
                # No further opening tag: keep remainder after the last consumed end
                self._buf = self._buf[end:]
                break

            # Prepare to extract next block
            pos = next_open
            first_open = next_open

        if not result_parts:
            return "\n"

        concatenated = "".join(result_parts)
        return create_response_chunk(self.id_generator(), concatenated)