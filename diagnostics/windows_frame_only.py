"""Time only the acquisition frame writes of performance/hdf5_replace.py, with none of its other measurements before them.

Usage, from the nionswift clone, with PYTHONPATH set to the clone, at each commit to compare:
    python windows_frame_only.py <path to hdf5_replace.py> <label>
It opens the script's project with a 64 x 64 image, then three times makes a new 4D item and writes the frames of its scan,
as the script's frame_write measurement does, timing each frame with the storage settling after it. It prints the median,
mean and p90 time of a frame write for each round. The script's frame_write comes after its reshape and rewrite
measurements; this shows whether the frame writes cost the same without them.
"""

import gc
import importlib.util
import pathlib
import statistics
import sys

ROUNDS = 3


def main() -> None:
    # the paths are made absolute first, since the script's setup changes the current directory.
    script_path, label = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]
    spec = importlib.util.spec_from_file_location("hdf5_replace", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(script_path), "--image-size", "64"]
    parser, arguments = module.parse_arguments()
    hdf5_replace = module.HDF5Replace(arguments, None)
    hdf5_replace.setup()
    try:
        hdf5_replace.open_write_project()
        for round_index in range(ROUNDS):
            hdf5_replace.begin_frames()
            gc.collect()
            times = sorted(hdf5_replace.write_frame(frame_index) for frame_index in range(arguments.scan_size ** 2))
            frames_written = hdf5_replace.end_frames()
            print(f"{label}  round {round_index + 1}  frame_write  median {statistics.median(times) * 1000:.3f} ms  mean {statistics.mean(times) * 1000:.3f} ms"
                  f"  p90 {times[int(len(times) * 0.9)] * 1000:.3f} ms  ({len(times)} frames, {frames_written} written)", flush=True)
    finally:
        hdf5_replace.teardown()


if __name__ == "__main__":
    main()
