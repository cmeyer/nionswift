"""Time only the acquisition frame writes of performance/hdf5_replace.py, with none of its other measurements before them.

Usage, from the nionswift clone, with PYTHONPATH set to the clone, at each commit to compare:
    python windows_frame_only.py <path to hdf5_replace.py> <label> [--rounds N] [--log <file>] [--no-sync] [--no-idle-close]
It opens the script's project with a 64 x 64 image, then for each round makes a new 4D item and writes the frames of its
scan, as the script's frame_write measurement does, timing each frame with the storage settling after it. It prints the
median, mean and p90 time of a frame write for each round. The script's frame_write comes after its reshape and rewrite
measurements; this shows whether the frame writes cost the same without them.

With --log, it writes the nionswift.data_files records into the file, with the time since the start and the thread of
each, and adds the start and end of each round, the median of each block of 128 frames, and the start and end of each
sync of a data file. On a version without that logger, the file holds only the driver's own records.

The data-handle branch syncs each pending file before it replaces the item's file, and closes a file once it has been idle
for a while. To find which of these slows later frame writes, --no-sync makes each sync do nothing, and --no-idle-close
keeps idle files open, as master does. Both patch the storage in this process only, and need the data-handle branch.
"""

import argparse
import gc
import importlib.util
import logging
import pathlib
import statistics
import sys
import time

BLOCK_FRAMES = 128


def main() -> None:
    parser = argparse.ArgumentParser(description="time only the frame writes of hdf5_replace.py")
    parser.add_argument("script", help="the path to hdf5_replace.py")
    parser.add_argument("label", help="a label for the results, such as a branch name")
    parser.add_argument("--rounds", type=int, default=3, help="the rounds, each with a new 4D item")
    parser.add_argument("--log", help="the file for the data files log")
    parser.add_argument("--no-sync", action="store_true", help="make each sync of a data file do nothing")
    parser.add_argument("--no-idle-close", action="store_true", help="keep idle data files open")
    options = parser.parse_args()
    # the paths are made absolute first, since the script's setup changes the current directory.
    script_path = pathlib.Path(options.script).resolve()
    log_path = pathlib.Path(options.log).resolve() if options.log else None
    driver_logger = logging.getLogger("frame_only")
    try:
        from nion.swift.model import HDF5Handler
    except ImportError:
        HDF5Handler = None  # type: ignore[assignment]
    is_branch = HDF5Handler is not None and hasattr(HDF5Handler, "_sync_file") and hasattr(HDF5Handler, "HDF5IdleFiles")
    if (options.no_sync or options.no_idle_close) and not is_branch:
        sys.exit("--no-sync and --no-idle-close need the data-handle branch checked out.")
    if options.no_sync:
        # the sync thread looks the function up in the module each time, so replacing it there changes every sync.
        setattr(HDF5Handler, "_sync_file", lambda file_path: None)
    if options.no_idle_close:
        setattr(HDF5Handler.HDF5IdleFiles, "close_idle_files", lambda self: False)
    if log_path:

        class ElapsedFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return f"{(record.created - start_wall) * 1000:10.1f} ms  {record.threadName:20}  {record.getMessage()}"

        start_wall = time.time()
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setFormatter(ElapsedFormatter())
        for logger in (logging.getLogger("nion.swift.data_files"), driver_logger):
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.propagate = False
        # the storage logs only a refused sync; this diagnostic wraps the module's sync function to record each one.
        if is_branch:
            sync_file = HDF5Handler._sync_file

            def logged_sync_file(file_path: pathlib.Path) -> None:
                driver_logger.debug("sync started: %s", file_path)
                try:
                    sync_file(file_path)
                finally:
                    driver_logger.debug("sync finished: %s", file_path)

            setattr(HDF5Handler, "_sync_file", logged_sync_file)
    spec = importlib.util.spec_from_file_location("hdf5_replace", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(script_path), "--image-size", "64"]
    script_parser, arguments = module.parse_arguments()
    hdf5_replace = module.HDF5Replace(arguments, None)
    hdf5_replace.setup()
    try:
        hdf5_replace.open_write_project()
        for round_index in range(options.rounds):
            hdf5_replace.begin_frames()
            gc.collect()
            driver_logger.debug("round %d frames started", round_index + 1)
            times = list[float]()
            for frame_index in range(arguments.scan_size ** 2):
                times.append(hdf5_replace.write_frame(frame_index))
                if len(times) % BLOCK_FRAMES == 0:
                    driver_logger.debug("round %d frames %d to %d: median %.3f ms", round_index + 1, len(times) - BLOCK_FRAMES + 1, len(times),
                                        statistics.median(times[-BLOCK_FRAMES:]) * 1000)
            driver_logger.debug("round %d frames finished", round_index + 1)
            frames_written = hdf5_replace.end_frames()
            driver_logger.debug("round %d ended", round_index + 1)
            times.sort()
            print(f"{options.label}  round {round_index + 1}  frame_write  median {statistics.median(times) * 1000:.3f} ms  mean {statistics.mean(times) * 1000:.3f} ms"
                  f"  p90 {times[int(len(times) * 0.9)] * 1000:.3f} ms  ({len(times)} frames, {frames_written} written)", flush=True)
    finally:
        hdf5_replace.teardown()


if __name__ == "__main__":
    main()
