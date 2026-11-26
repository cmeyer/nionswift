"""Sample every thread's stack during the timed steps of performance/hdf5_replace.py, and report where each thread spends its time.

Usage, from the nionswift clone, with PYTHONPATH set to the clone:
    python windows_thread_sampler.py <path to hdf5_replace.py> <output file> <label>
It runs the script's measurements in the script's own order. A sampler thread records the frames of every other thread
every millisecond while a timed step runs, which the script marks by enabling its profiler, and writes, for each
measurement and each thread, the functions which appear most often innermost in its samples and the stacks which hold it
most often. The main thread's samples show where its time goes, waits included, which a profile doesn't separate from
other threads' waits.
"""

import collections
import gc
import importlib.util
import pathlib
import sys
import threading
import time
import traceback

DEPTH = 12
INTERVAL_S = 0.001


def main() -> None:
    # the paths are made absolute first, since the script's setup changes the current directory.
    script_path, output_path, label = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve(), sys.argv[3]
    spec = importlib.util.spec_from_file_location("hdf5_replace", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(script_path)]
    parser, arguments = module.parse_arguments()
    # measurement -> thread name -> stack text or innermost function -> count, and measurement -> thread name -> count.
    stacks = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    tops = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    sample_counts = collections.defaultdict(collections.Counter)
    measurement = [""]
    sampling = threading.Event()
    stop = threading.Event()
    sampler_ident: list[int] = []

    class TimedSteps:
        # the script enables its profiler for each timed step and disables it after; this samples during them instead.
        def enable(self) -> None:
            sampling.set()

        def disable(self) -> None:
            sampling.clear()

    def sample() -> None:
        sampler_ident.append(threading.get_ident())
        names: dict[int | None, str] = dict()
        while not stop.is_set():
            if sampling.is_set():
                names.update({thread.ident: thread.name for thread in threading.enumerate()})
                for ident, frame in sys._current_frames().items():
                    if ident == sampler_ident[0]:
                        continue
                    name = names.get(ident, str(ident))
                    stack = traceback.extract_stack(frame)[-DEPTH:]
                    current = measurement[0]
                    sample_counts[current][name] += 1
                    tops[current][name][f"{stack[-1].name} ({pathlib.Path(stack[-1].filename).name}:{stack[-1].lineno})"] += 1
                    stacks[current][name][" <- ".join(f"{entry.name} ({pathlib.Path(entry.filename).name}:{entry.lineno})" for entry in reversed(stack))] += 1
                # a frame keeps its locals, such as a data handle, which would keep a pending file from its replace.
                frame = None
            time.sleep(INTERVAL_S)

    hdf5_replace = module.HDF5Replace(arguments, TimedSteps())
    times = dict[str, list[float]]()

    def run(name: str, work: list) -> None:
        measurement[0] = name
        times[name] = list(work)

    sampler = threading.Thread(target=sample, name="sampler", daemon=True)
    sampler.start()
    hdf5_replace.setup()
    try:
        hdf5_replace.open_write_project()
        gc.collect()
        # the measurements run in the script's own order, since each starts from the state the one before leaves. each
        # list is made inside run, so that its samples are counted under its name.
        samples = arguments.samples
        measurement[0] = "rewrite"
        run("rewrite", [hdf5_replace.rewrite("rewrite") for _ in range(samples)])
        measurement[0] = "reshape"
        run("reshape", [hdf5_replace.reshape(index, False) for index in range(samples)])
        measurement[0] = "reshape_held"
        run("reshape_held", [hdf5_replace.reshape(samples + index, True) for index in range(samples)])
        measurement[0] = "rewrite_after_reshape"
        run("rewrite_after_reshape", [hdf5_replace.rewrite_after_reshape(2 * samples + index) for index in range(samples)])
        measurement[0] = "begin_frames"
        hdf5_replace.begin_frames()
        gc.collect()
        measurement[0] = "frame_write"
        run("frame_write", [hdf5_replace.write_frame(index) for index in range(arguments.scan_size ** 2)])
        measurement[0] = "end_frames"
        hdf5_replace.end_frames()
        measurement[0] = "reserve"
        run("reserve", [hdf5_replace.reserve() for _ in range(samples)])
        measurement[0] = "open_load_project"
        hdf5_replace.open_load_project()
        gc.collect()
        measurement[0] = "load_read"
        run("load_read", [hdf5_replace.load_read() for _ in range(arguments.loads)])
    finally:
        stop.set()
        sampler.join()
        hdf5_replace.teardown()
    lines = [f"{label}: sampled every {INTERVAL_S * 1000:.0f} ms during the timed steps"]
    for name, work_times in times.items():
        total = sum(work_times) * 1000
        lines.append(f"\n######## {name}: {len(work_times)} samples, {total:.1f} ms timed in all, median {sorted(work_times)[len(work_times) // 2] * 1000:.3f} ms")
        for thread_name, count in sample_counts[name].most_common():
            lines.append(f"\n== thread {thread_name}: {count} samples")
            lines.append("innermost functions:")
            lines.extend(f"  {n:6d}  {function}" for function, n in tops[name][thread_name].most_common(15))
            lines.append("stacks:")
            lines.extend(f"  {n:6d}  {stack}" for stack, n in stacks[name][thread_name].most_common(10))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{label}: sampled " + ", ".join(f"{name} {sum(sample_counts[name].values())}" for name in times))


if __name__ == "__main__":
    main()
