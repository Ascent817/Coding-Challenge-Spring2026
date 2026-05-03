from __future__ import annotations

import ctypes
from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):

    _NO_READER = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        if reader != self._NO_READER and not (0 <= reader < num_readers):
            raise ValueError("reader index out of range")
        if cache_align and (cache_size < 1 or (cache_size & (cache_size - 1)) != 0):
            raise ValueError("cache_size must be a power of two")

        slots_per_reader = 3
        metadata_slots = 6
        header_slots = metadata_slots + num_readers * slots_per_reader
        header_bytes = header_slots * ctypes.sizeof(ctypes.c_int64)

        super().__init__(name=name, create=create, size=header_bytes + size)

        if self.buf is None:
            raise RuntimeError("shared memory buffer is unavailable")

        self.header = (ctypes.c_int64 * header_slots).from_buffer(self.buf, 0)

        self.buffer_size = size
        self.num_readers = num_readers
        self.reader = reader
        self._slots_per_reader = slots_per_reader
        self._metadata_slots = metadata_slots

        self._payload = self.buf[header_bytes:]

        if create:
            self.header[0] = 0
            self.header[1] = size
            self.header[2] = num_readers
            for i in range(num_readers):
                slot = metadata_slots + i * slots_per_reader
                self.header[slot] = 0
                self.header[slot + 1] = 0
                self.header[slot + 2] = 0

        self._cached_max_writable = size


    def close(self) -> None:
        """
        Release local views and close this process's handle to the shared memory.

        This should not destroy the buffer for other attached processes.
        """
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        """
        Enter the context manager.

        Reader instances are expected to mark themselves active while inside the
        context. Writer-only instances can simply return `self`.
        """
        return self

    def __exit__(self, *_):
        """
        Exit the context manager.

        Reader instances are expected to mark themselves inactive on exit, then
        close local resources.
        """
        self.close()

    def calculate_pressure(self) -> int:
        """
        Return current writer pressure as an integer percentage.

        Pressure is based on how much of the bounded storage is currently in use
        relative to the slowest active reader.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.calculate_pressure")

    def int_to_pos(self, value: int) -> int:
        """
        Convert an absolute position counter into a position inside the bounded payload area.

        If your design does not use modulo arithmetic internally, you may still
        keep this helper as the mapping from logical positions to buffer offsets.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.int_to_pos")

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """
        Store this reader's absolute read position in shared state.

        This must fail clearly when called on a writer-only instance.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.update_reader_pos")

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.set_reader_active")

    def is_reader_active(self) -> bool:
        """
        Return whether this reader is currently marked active.

        This must fail clearly when called on a writer-only instance.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.is_reader_active")

    def update_write_pos(self, new_writer_pos: int) -> None:
        """
        Store the writer's absolute write position in shared state.

        The write position is what makes newly written bytes visible to readers.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.update_write_pos")

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.inc_writer_pos")

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.inc_reader_pos")

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.get_write_pos")

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """
        Return how many bytes the writer can safely expose right now.

        This should take active readers into account. `force_rescan=True` is used
        by the tests to ensure externally updated reader positions are observed.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.compute_max_amount_writable")

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.jump_to_writer")

    def expose_writer_mem_view(self, size: int) -> RingView:
        """
        Return a writable view tuple for up to `size` bytes.

        The return shape is:
        - `mv1`: first writable view
        - `mv2`: optional second writable view if the exposed region is split
        - `actual_size`: how many bytes are actually writable right now
        - `split`: whether the writable region is split across two views

        If less than `size` bytes are currently writable, clamp to the amount
        available rather than raising.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.expose_writer_mem_view")

    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a readable view tuple for up to `size` bytes.

        The shape matches `expose_writer_mem_view()`. If less than `size` bytes
        are currently readable, clamp to the amount available rather than raising.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.expose_reader_mem_view")

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy bytes from `src` into the exposed writer view(s).

        If `src` is larger than the destination region, copy only the prefix that fits.
        This helper should not publish data by itself; publishing happens when the
        writer position is advanced.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.simple_write")

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.simple_read")

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.write_array")

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.read_array")
