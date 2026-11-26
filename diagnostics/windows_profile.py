"""Profile each measurement of performance/hdf5_replace.py on its own, and save one profile per measurement.

Usage, from the nionswift clone, with PYTHONPATH set to the clone:
    python windows_profile.py <path to hdf5_replace.py> <output folder> <label>
It runs the script's standard case once, with a profiler which covers only the timed steps, as the script's --profile does,
and writes <label>-<measurement>.prof and <label>-<measurement>.txt (the 30 functions taking the most time, by own time and by
cumulative time) for each measurement. The profiler inflates the times, so the times it prints are for comparing the two
versions' profiles, not for the table.
"""

import cProfile
import gc
import importlib.util
import io
import pathlib
import pstats
import statistics
import sys


def main() -> None:
    # the paths are made absolute first, since the script's setup changes the current directory.
    script_path, output_path, label = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve(), sys.argv[3]
    output_path.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("hdf5_replace", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(script_path), "--profile"]
    parser, arguments = module.parse_arguments()
    profiler = cProfile.Profile()
    hdf5_replace = module.HDF5Replace(arguments, profiler)

    def save(measurement_name: str, times: list[float]) -> None:
        profiler.dump_stats(output_path / f"{label}-{measurement_name}.prof")
        text = io.StringIO()
        stats = pstats.Stats(profiler, stream=text).strip_dirs()
        text.write(f"{measurement_name}: {len(times)} samples, profiled mean {statistics.mean(times) * 1000:.3f} ms, median {statistics.median(times) * 1000:.3f} ms\n")
        stats.sort_stats("tottime").print_stats(30)
        stats.sort_stats("cumulative").print_stats(30)
        (output_path / f"{label}-{measurement_name}.txt").write_text(text.getvalue(), encoding="utf-8")
        print(f"{measurement_name}: profiled mean {statistics.mean(times) * 1000:.3f} ms", flush=True)
        profiler.clear()

    hdf5_replace.setup()
    try:
        hdf5_replace.open_write_project()
        gc.collect()
        # the measurements run in the script's own order, since each starts from the state the one before leaves.
        save("rewrite", [hdf5_replace.rewrite("rewrite") for _ in range(arguments.samples)])
        save("reshape", [hdf5_replace.reshape(sample_index, False) for sample_index in range(arguments.samples)])
        save("reshape_held", [hdf5_replace.reshape(arguments.samples + sample_index, True) for sample_index in range(arguments.samples)])
        save("rewrite_after_reshape", [hdf5_replace.rewrite_after_reshape(2 * arguments.samples + sample_index) for sample_index in range(arguments.samples)])
        hdf5_replace.begin_frames()
        gc.collect()
        save("frame_write", [hdf5_replace.write_frame(frame_index) for frame_index in range(arguments.scan_size ** 2)])
        hdf5_replace.end_frames()
        save("reserve", [hdf5_replace.reserve() for _ in range(arguments.samples)])
        hdf5_replace.open_load_project()
        gc.collect()
        save("load_read", [hdf5_replace.load_read() for _ in range(arguments.loads)])
    finally:
        hdf5_replace.teardown()


if __name__ == "__main__":
    main()
