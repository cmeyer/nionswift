"""Time acquisition frame writes through the checked out nionswift's HDF5 storage handler and through plain h5py, in several kinds of file.

Usage, from the nionswift clone, with PYTHONPATH set to the clone, at each commit to compare:
    python windows_handler_probe.py <label>
Each case makes a new file in the system temporary folder, reserves a 32 x 32 scan of 256 x 256 float32 frames with the
chunks nionswift gives it, and times the write of each of the 1024 frames, as an acquisition writes them during a
transaction, which delays the properties. The cases run in turn, three rounds, so that drift affects them alike:
- handler, item file first: the handler writes the item's properties first, which makes the file with an empty dataset,
  as adding a data item does, and then reserves. On the data-handle branch the reserve goes to a pending file.
- handler, no item file: the handler reserves into a new file. On the data-handle branch the reserve goes into the item's
  file, without a pending file.
- h5py, item name: plain h5py, into a file named as a data item's file.
- h5py, pending name: plain h5py, into a file named as a pending file, with a leading dot.
It prints the median and mean time of a frame write for each case and round, then removes its files.
"""

import datetime
import pathlib
import platform
import statistics
import sys
import tempfile
import time
import typing

import h5py
import numpy

from nion.data import DataAndMetadata
from nion.swift.model import HDF5Handler

SCAN = 32
FRAME = 256
ROUNDS = 3
DATA_SHAPE = (SCAN, SCAN, FRAME, FRAME)
DATA_DESCRIPTOR = DataAndMetadata.DataDescriptor(False, 2, 2)


def time_frames(data: typing.Any) -> list[float]:
    """Return the time in seconds of each frame write into data, a dataset or a data handle, as the data item writes one."""
    frame = numpy.ones((1, 1, FRAME, FRAME), numpy.float32)
    times = list[float]()
    for frame_index in range(SCAN * SCAN):
        scan_y, scan_x = divmod(frame_index, SCAN)
        key = (slice(scan_y, scan_y + 1), slice(scan_x, scan_x + 1), slice(None), slice(None))
        start = time.perf_counter()
        data[key] = frame
        times.append(time.perf_counter() - start)
    return times


def handler_frames(file_path: pathlib.Path, is_item_file_first: bool) -> list[float]:
    """Return the frame write times through a storage handler for the file at file_path."""
    handler = HDF5Handler.HDF5Handler(file_path)
    try:
        now = datetime.datetime.now()
        if is_item_file_first:
            handler.write_properties({"uuid": "00000000-0000-0000-0000-000000000000", "version": 16}, now)
        handler.reserve_data(DATA_SHAPE, numpy.float32, DATA_DESCRIPTOR, now)
        data = handler.read_data()
        assert data is not None
        times = time_frames(data)
        # the data and its handle go before the handler closes, as a data item lets go of them.
        del data
        return times
    finally:
        handler.close()


def h5py_frames(file_path: pathlib.Path) -> list[float]:
    """Return the frame write times through plain h5py for a new file at file_path."""
    chunks = HDF5Handler.get_write_chunk_shape_for_data(DATA_SHAPE, numpy.float32, DATA_DESCRIPTOR)
    with h5py.File(file_path, "a") as file:
        dataset = file.require_dataset("data", shape=DATA_SHAPE, dtype=numpy.float32, fillvalue=0, chunks=chunks)
        return time_frames(dataset)


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"{label}: python {platform.python_version()}, h5py {h5py.__version__}, HDF5 {h5py.version.hdf5_version}, {sys.platform}, "
          f"chunks {HDF5Handler.get_write_chunk_shape_for_data(DATA_SHAPE, numpy.float32, DATA_DESCRIPTOR)}", flush=True)
    cases: list[tuple[str, typing.Callable[[pathlib.Path, int], list[float]]]] = [
        ("handler, item file first", lambda folder, index: handler_frames(folder / f"data_first{index}.h5", True)),
        ("handler, no item file", lambda folder, index: handler_frames(folder / f"data_none{index}.h5", False)),
        ("h5py, item name", lambda folder, index: h5py_frames(folder / f"data_plain{index}.h5")),
        ("h5py, pending name", lambda folder, index: h5py_frames(folder / f".data_plain{index}.1.h5")),
    ]
    with tempfile.TemporaryDirectory() as directory:
        folder = pathlib.Path(directory)
        for round_index in range(ROUNDS):
            for name, run in cases:
                times = run(folder, round_index)
                print(f"{label}  round {round_index + 1}  {name:26}  median {statistics.median(times) * 1000:.3f} ms  mean {statistics.mean(times) * 1000:.3f} ms",
                      flush=True)
        files = sorted(path.name for path in folder.iterdir())
        # a hidden file here is a pending file which a handler left.
        print(f"{label}: files in the folder before it is removed: {', '.join(files) if files else 'none'}")


if __name__ == "__main__":
    main()
