"""Tests for seek correctness in StreamableSourceWrapper and BufferedIOBaseWrapper.

Regression tests for #2910: relative seeks (SEEK_CUR/SEEK_END) were silently
ignored, causing metadata parsing to walk the entire stream and time out (~10 s).
"""

import io

import miniaudio
import pytest

from pyatv.protocols.raop.audio_source import (
    HEADROOM_SIZE,
    BufferedIOBaseWrapper,
    StreamableSourceWrapper,
)
from pyatv.support.buffer import SemiSeekableBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStreamableSource(miniaudio.StreamableSource):
    """Minimal StreamableSource backed by a SemiSeekableBuffer for unit tests."""

    def __init__(self, buffer: SemiSeekableBuffer) -> None:
        self._buf = buffer

    def read(self, num_bytes: int) -> bytes:
        """Read from the backing buffer."""
        return self._buf.get(num_bytes)

    def seek(self, offset: int, origin: miniaudio.SeekOrigin) -> bool:
        """Only SEEK_SET is supported; everything else returns False."""
        if origin == miniaudio.SeekOrigin.START:
            return self._buf.seek(offset)
        return False


def _make_ssw(data: bytes, headroom: int = HEADROOM_SIZE) -> StreamableSourceWrapper:
    """Build a StreamableSourceWrapper over *data* with the given *headroom*."""
    buf = SemiSeekableBuffer(
        max(len(data), headroom) + 64,
        seekable_headroom=headroom,
        protected_headroom=True,
    )
    buf.add(data)
    src = _FakeStreamableSource(buf)
    return StreamableSourceWrapper(src, buf)


def _make_biow(data: bytes, headroom: int = HEADROOM_SIZE) -> BufferedIOBaseWrapper:
    """Build a BufferedIOBaseWrapper over *data* with the given *headroom*."""
    buf = SemiSeekableBuffer(
        max(len(data), headroom) + 64,
        seekable_headroom=headroom,
        protected_headroom=True,
    )
    buf.add(data)
    return BufferedIOBaseWrapper(io.BytesIO(data), buf)


# ---------------------------------------------------------------------------
# StreamableSourceWrapper seek tests
# ---------------------------------------------------------------------------


class TestStreamableSourceWrapperSeek:
    """StreamableSourceWrapper.seek must handle all io.SEEK_* constants correctly."""

    def test_seek_set_to_zero(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        assert wrapper.seek(0, io.SEEK_SET) == 0

    def test_seek_set_within_headroom(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(80)
        assert wrapper.seek(40, io.SEEK_SET) == 40

    def test_seek_cur_zero_is_noop(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(50)
        assert wrapper.seek(0, io.SEEK_CUR) == 50

    def test_seek_cur_forward(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(30)
        result = wrapper.seek(20, io.SEEK_CUR)
        assert result == 50

    def test_seek_cur_backward(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(60)
        result = wrapper.seek(-20, io.SEEK_CUR)
        assert result == 40

    def test_seek_end_raises(self):
        """SEEK_END is unsupported on a streaming source and must raise OSError."""
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        with pytest.raises(OSError):
            wrapper.seek(0, io.SEEK_END)

    def test_seek_cur_beyond_headroom_raises(self):
        """SEEK_CUR to an absolute position past the headroom must raise."""
        headroom = 64
        buf = SemiSeekableBuffer(
            256,
            seekable_headroom=headroom,
            protected_headroom=False,
        )
        buf.add(b"x" * 200)
        src = _FakeStreamableSource(buf)
        wrapper = StreamableSourceWrapper(src, buf)
        wrapper.read(headroom)  # exhausts and discards headroom (unprotected)
        with pytest.raises(OSError):
            wrapper.seek(-1, io.SEEK_CUR)  # absolute = headroom - 1; headroom gone


# ---------------------------------------------------------------------------
# BufferedIOBaseWrapper seek tests
# ---------------------------------------------------------------------------


class TestBufferedIOBaseWrapperSeek:
    """BufferedIOBaseWrapper.seek must handle all io.SEEK_* constants correctly."""

    def test_seek_set_to_zero(self):
        wrapper = _make_biow(b"x" * 200, headroom=128)
        assert wrapper.seek(0, io.SEEK_SET) == 0

    def test_seek_cur_zero_is_noop(self):
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(50)
        assert wrapper.seek(0, io.SEEK_CUR) == 50

    def test_seek_cur_forward(self):
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(30)
        result = wrapper.seek(20, io.SEEK_CUR)
        assert result == 50

    def test_seek_cur_backward(self):
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(60)
        result = wrapper.seek(-20, io.SEEK_CUR)
        assert result == 40

    def test_seek_end_raises(self):
        """SEEK_END is unsupported and must raise OSError."""
        wrapper = _make_biow(b"x" * 200, headroom=128)
        with pytest.raises(OSError):
            wrapper.seek(0, io.SEEK_END)

    def test_seek_cur_beyond_headroom_raises(self):
        """SEEK_CUR to an absolute position past the headroom must raise."""
        headroom = 64
        buf = SemiSeekableBuffer(
            256,
            seekable_headroom=headroom,
            protected_headroom=False,
        )
        buf.add(b"x" * 200)
        wrapper = BufferedIOBaseWrapper(io.BytesIO(b"x" * 200), buf)
        wrapper.read(headroom)  # exhausts and discards headroom (unprotected)
        with pytest.raises(OSError):
            wrapper.seek(-1, io.SEEK_CUR)  # absolute = headroom - 1; headroom gone
