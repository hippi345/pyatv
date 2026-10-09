"""Tests for seek correctness in StreamableSourceWrapper and BufferedIOBaseWrapper.

Regression tests for #2910: relative seeks (SEEK_CUR/SEEK_END) were silently
ignored, causing metadata parsing to walk the entire stream and time out (~10 s).
"""

import asyncio
import io
import struct
import time

import miniaudio
import pytest

from pyatv.protocols.raop.audio_source import (
    BUFFER_SIZE,
    HEADROOM_SIZE,
    BufferedIOBaseWrapper,
    PatchedIceCastClient,
    StreamableSourceWrapper,
    get_buffered_io_metadata,
)
from pyatv.support.buffer import SemiSeekableBuffer
from pyatv.support.metadata import EMPTY_METADATA

from tests.utils import data_path

pytestmark = pytest.mark.asyncio


def _make_large_wav(total_bytes: int = BUFFER_SIZE + 8192) -> bytes:
    """Return a valid WAV file of *total_bytes* filled with silence.

    Using real WAV structure ensures TinyTag can begin parsing (and thus
    exercises its chunk-walking loop) rather than giving up immediately with
    UnsupportedFormatError (which would happen with a buffer of all zeros).
    """
    data_size = total_bytes - 44
    return (
        struct.pack("<4sI4s", b"RIFF", data_size + 36, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 2, 44100, 176400, 4, 16)
        + struct.pack("<4sI", b"data", data_size)
        + b"\x00" * data_size
    )


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

    async def test_seek_set_to_zero(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        assert wrapper.seek(0, io.SEEK_SET) == 0

    async def test_seek_set_within_headroom(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(80)
        assert wrapper.seek(40, io.SEEK_SET) == 40

    async def test_seek_cur_zero_is_noop(self):
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(50)
        assert wrapper.seek(0, io.SEEK_CUR) == 50

    async def test_seek_cur_forward(self):
        """SEEK_CUR with a positive offset advances the cursor."""
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(30)
        result = wrapper.seek(20, io.SEEK_CUR)
        assert result == 50

    async def test_seek_cur_backward(self):
        """SEEK_CUR with a negative offset moves the cursor back."""
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(60)
        result = wrapper.seek(-20, io.SEEK_CUR)
        assert result == 40

    async def test_seek_end_returns_buffered_size(self):
        """seek(0, SEEK_END) returns total buffered bytes so callers learn the size.

        tell() must also return that value so that callers (e.g. TinyTag) which
        read the position via tell() after the seek get the correct file size.
        The invariant total = buffer.position + buffer.size must hold at any
        cursor position, so we verify after an initial read advances position.
        """
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        wrapper.read(50)  # advance position to 50; size drops to 150
        result = wrapper.seek(0, io.SEEK_END)
        assert result == 200  # total must still equal original buffered size
        assert wrapper.tell() == 200

    async def test_seek_end_past_buffer_raises(self):
        """Seeking past the end of buffered data (pos > 0) must raise OSError."""
        wrapper = _make_ssw(b"x" * 200, headroom=128)
        with pytest.raises(OSError):
            wrapper.seek(1, io.SEEK_END)

    async def test_seek_cur_beyond_headroom_raises(self):
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

    async def test_seek_set_to_zero(self):
        """SEEK_SET 0 returns 0 and resets the position."""
        wrapper = _make_biow(b"x" * 200, headroom=128)
        assert wrapper.seek(0, io.SEEK_SET) == 0

    async def test_seek_cur_zero_is_noop(self):
        """SEEK_CUR with offset 0 returns the current position unchanged."""
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(50)
        assert wrapper.seek(0, io.SEEK_CUR) == 50

    async def test_seek_cur_forward(self):
        """SEEK_CUR with a positive offset advances the cursor."""
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(30)
        result = wrapper.seek(20, io.SEEK_CUR)
        assert result == 50

    async def test_seek_cur_backward(self):
        """SEEK_CUR with a negative offset moves the cursor back."""
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(60)
        result = wrapper.seek(-20, io.SEEK_CUR)
        assert result == 40

    async def test_seek_end_returns_buffered_size(self):
        """seek(0, SEEK_END) returns total buffered bytes.

        tell() must also return that value so that callers (e.g. TinyTag) which
        read the position via tell() after the seek get the correct file size.
        The invariant total = buffer.position + buffer.size must hold at any
        cursor position, so we verify after an initial read advances position.
        """
        wrapper = _make_biow(b"x" * 200, headroom=128)
        wrapper.read(50)  # advance position to 50; size drops to 150
        result = wrapper.seek(0, io.SEEK_END)
        assert result == 200  # total must still equal original buffered size
        assert wrapper.tell() == 200

    async def test_seek_end_past_buffer_raises(self):
        """Seeking past end (pos > 0) must raise OSError."""
        wrapper = _make_biow(b"x" * 200, headroom=128)
        with pytest.raises(OSError):
            wrapper.seek(1, io.SEEK_END)

    async def test_seek_cur_beyond_headroom_raises(self):
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


# ---------------------------------------------------------------------------
# Metadata integration tests (#2910)
# ---------------------------------------------------------------------------


async def test_metadata_small_buffered_stream_preserved():
    """Metadata from a small, fully-buffered stream is returned correctly.

    Verifies that the SEEK_END implementation returns the correct file size so
    that TinyTag can navigate to the tag position, rather than raising
    immediately and losing metadata for small sources.

    Run this test against master (reverted code) to observe: on master TinyTag
    still finds the metadata via sequential read (SEEK_CUR ignored → falls back
    to byte-by-byte chunk scan), so this test PASSES on both master and fix.
    It guards against a naïve "always raise on SEEK_END" regression.
    """
    with open(data_path("audio_1_packet_metadata.wav"), "rb") as fh:
        audio_data = fh.read()

    assert (
        len(audio_data) < HEADROOM_SIZE
    ), "fixture must fit within seekable headroom so SEEK_END can seek there"

    buf = SemiSeekableBuffer(
        BUFFER_SIZE,
        seekable_headroom=HEADROOM_SIZE,
        protected_headroom=True,
    )
    buf.add(audio_data)
    src = _FakeStreamableSource(buf)
    wrapper = StreamableSourceWrapper(src, buf)

    metadata = await get_buffered_io_metadata(wrapper)

    assert metadata.title == "pyatv"
    assert metadata.artist == "postlund"


async def test_metadata_large_stream_returns_fast(monkeypatch):
    """Metadata parsing for a stream larger than BUFFER_SIZE returns in < 2 s.

    Before the fix, SEEK_CUR was silently ignored so TinyTag walked the stream
    byte-by-byte, exhausted the pre-buffer, and PatchedIceCastClient.read
    blocked for DEFAULT_TIMEOUT (~10 s) before returning EMPTY_METADATA.

    With the fix, SEEK_END returns the buffered size immediately.  TinyTag
    either navigates within the buffered window (no blocking reads) or gives
    up via OSError when a seek falls outside the headroom — either path
    completes well under 2 s.

    Revert check: on master this assertion fails with elapsed ~ DEFAULT_TIMEOUT.
    """
    large_body = _make_large_wav(BUFFER_SIZE + 8192)

    class _FakeRaw:
        """Minimal raw response body that replays *large_body* sequentially."""

        def __init__(self) -> None:
            """Set up a sequential reader over the pre-built WAV data."""
            self._data = large_body
            self._pos = 0

        def read(self, n: int) -> bytes:
            """Return the next *n* bytes and advance the internal cursor."""
            chunk = self._data[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    class _FakeHandle:
        status_code = 200
        reason = "OK"
        headers: dict = {}

        def __init__(self) -> None:
            self.raw = _FakeRaw()

        def __enter__(self) -> "_FakeHandle":
            return self

        def __exit__(self, *exc) -> bool:
            """Context manager exit; no cleanup needed."""
            return False

    monkeypatch.setattr(
        "pyatv.protocols.raop.audio_source.requests.get",
        lambda url, stream, timeout: _FakeHandle(),
    )

    loop = asyncio.get_event_loop()
    buf = SemiSeekableBuffer(
        BUFFER_SIZE,
        seekable_headroom=HEADROOM_SIZE,
        protected_headroom=True,
    )
    source = await loop.run_in_executor(None, PatchedIceCastClient, buf, "http://test")
    wrapper = StreamableSourceWrapper(source, buf)

    start = time.monotonic()
    metadata = await get_buffered_io_metadata(wrapper)
    elapsed = time.monotonic() - start

    await loop.run_in_executor(None, source.close)

    assert elapsed < 2.0, (
        f"metadata parsing took {elapsed:.1f} s — expected < 2 s. "
        "On unfixed code SEEK_CUR is ignored so TinyTag walks the entire "
        "stream, exhausting the buffer and hitting DEFAULT_TIMEOUT (~10 s)."
    )
    assert metadata.title is None
    assert metadata.artist is None


async def test_metadata_restore_failure_does_not_raise(caplog):
    """A failed position-restore in get_buffered_io_metadata must not raise.

    After the SEEK_CUR/SEEK_END fix, seek() raises OSError on failure instead
    of silently ignoring it.  The finally-block in get_buffered_io_metadata
    restores the stream position after parsing; if that seek fails the function
    must swallow the OSError, log a warning, and return normally so that
    playback is never interrupted by a metadata position-restore failure.
    """
    import logging

    with open(data_path("audio_1_packet_metadata.wav"), "rb") as fh:
        audio_data = fh.read()

    buf = SemiSeekableBuffer(
        BUFFER_SIZE,
        seekable_headroom=HEADROOM_SIZE,
        protected_headroom=True,
    )
    buf.add(audio_data)
    src = _FakeStreamableSource(buf)
    wrapper = StreamableSourceWrapper(src, buf)

    # Replace seek with a version that always raises OSError.  We install it
    # after constructing the wrapper so that get_buffered_io_metadata's initial
    # seek(0) (which happens *before* get_metadata is called) still works; the
    # broken seek only fires inside the finally-block restore.
    # To achieve this without relying on TinyTag's exact call count we patch
    # StreamableSourceWrapper.seek on the class, gated by a flag that is set
    # when TinyTag has finished and the finally-block is entered.
    restore_phase = {"active": False}
    original_seek = type(wrapper).seek

    def failing_seek(self, pos, origin=io.SEEK_SET):
        """Raise OSError during the restore phase, pass through otherwise."""
        if restore_phase["active"]:
            raise OSError("simulated restore failure")
        return original_seek(self, pos, origin)

    type(wrapper).seek = failing_seek
    try:
        # Trigger the restore phase just before the finally-block runs by
        # monkey-patching get_metadata to set the flag after it returns.
        import pyatv.protocols.raop.audio_source as _mod

        real_get_metadata = _mod.get_metadata

        async def _patched_get_metadata(buf_arg):
            result = await real_get_metadata(buf_arg)
            restore_phase["active"] = True
            return result

        _mod.get_metadata = _patched_get_metadata
        try:
            with caplog.at_level(
                logging.WARNING, logger="pyatv.protocols.raop.audio_source"
            ):
                result = await get_buffered_io_metadata(wrapper)
        finally:
            _mod.get_metadata = real_get_metadata
    finally:
        type(wrapper).seek = original_seek

    assert isinstance(result, type(EMPTY_METADATA))
    assert any("Failed to restore position" in r.message for r in caplog.records)
