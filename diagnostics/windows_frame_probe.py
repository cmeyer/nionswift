"""Time single-frame writes into an HDF5 dataset several ways, and check that a read-only file refuses the open of a sync.

Usage, from the nionswift clone with the data-handle branch checked out and PYTHONPATH set to the clone:
    python windows_frame_probe.py
It needs no quiet machine for the refusal check, but the timings mean most on a quiet one. It writes into the system
temporary folder and removes what it writes.

The frame writes separate the parts of the branch's acquisition path which could cost more on Windows: h5py's write
alone, the write through the branch's data handle (a lock and a counter around h5py's write), a flush after each write,
which the storage makes after a partial update, and the same while another thread reads the dataset, as a display does.
"""

import os
import pathlib
import platform
import stat
import statistics
import sys
import tempfile
import threading
import time
import typing

import h5py
import numpy

try:
    from nion.swift.model import DataHandle
except ImportError:
    DataHandle = None  # type: ignore[assignment]

SCAN = 32
FRAME = 256


def time_writes(write: typing.Callable[[int], None]) -> tuple[float, float]:
    """Return the median and mean time in milliseconds of writing every frame of the scan once."""
    times = list[float]()
    for frame_index in range(SCAN * SCAN):
        start = time.perf_counter()
        write(frame_index)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000, statistics.mean(times) * 1000


def main() -> None:
    print(f"python {platform.python_version()}, h5py {h5py.__version__}, HDF5 {h5py.version.hdf5_version}, {sys.platform}, "
          f"data handle {'available' if DataHandle else 'not available (not the data-handle branch)'}")
    frame = numpy.ones((FRAME, FRAME), numpy.float32)
    with tempfile.TemporaryDirectory() as directory:
        file_path = pathlib.Path(directory) / "frames.h5"
        with h5py.File(file_path, "a") as f:
            dataset = f.require_dataset("data", shape=(SCAN, SCAN, FRAME, FRAME), dtype=numpy.float32, fillvalue=0)
            cases: list[tuple[str, typing.Callable[[int], None]]] = [
                ("h5py write", lambda i: dataset.__setitem__(numpy.unravel_index(i, (SCAN, SCAN)), frame)),
                ("h5py write and flush", lambda i: (dataset.__setitem__(numpy.unravel_index(i, (SCAN, SCAN)), frame), f.flush())),
            ]
            if DataHandle:
                data_handle = DataHandle.DataHandle(dataset, DataHandle.DataHandleWriteGeneration())
                cases.append(("data handle write", lambda i: data_handle.__setitem__(numpy.unravel_index(i, (SCAN, SCAN)), frame)))
                cases.append(("data handle write and flush", lambda i: (data_handle.__setitem__(numpy.unravel_index(i, (SCAN, SCAN)), frame), f.flush())))
            # each case runs twice, alternating, so that drift affects all of them alike.
            for round_index in range(2):
                for name, write in cases:
                    median_ms, mean_ms = time_writes(write)
                    print(f"round {round_index + 1}  {name:28}  median {median_ms:.3f} ms  mean {mean_ms:.3f} ms")
            # another thread reads frames while this one writes, as a display reads the latest frame.
            stop = threading.Event()

            def read_frames() -> None:
                while not stop.is_set():
                    numpy.asarray(dataset[0, 0])

            reader = threading.Thread(target=read_frames, daemon=True)
            reader.start()
            try:
                for name, write in cases:
                    median_ms, mean_ms = time_writes(write)
                    print(f"with reader  {name:28}  median {median_ms:.3f} ms  mean {mean_ms:.3f} ms")
            finally:
                stop.set()
                reader.join()
        # the sync opens a pending file with open(path, "r+b"); a read-only file must refuse that open for the refused sync test to prove its case.
        os.chmod(file_path, stat.S_IREAD)
        try:
            with open(file_path, "r+b"):
                pass
            print("read-only file: open r+b succeeded, so chmod doesn't refuse the sync's open")
        except OSError as e:
            print(f"read-only file: open r+b raised {type(e).__name__}, errno {e.errno}, winerror {getattr(e, 'winerror', None)}")
        finally:
            os.chmod(file_path, stat.S_IREAD | stat.S_IWRITE)


if __name__ == "__main__":
    main()
