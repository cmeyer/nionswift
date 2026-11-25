"""Sample every thread's stack while performance/hdf5_replace.py runs its rewrite measurement, and report where each thread spends its time.

Usage, from the nionswift clone, with PYTHONPATH set to the clone:
    python windows_thread_sampler.py <path to hdf5_replace.py> <output file> <label>
It runs the script's standard setup, then 10 rewrites with a sampler thread which records the innermost frames of every
other thread every millisecond, and writes, for each thread, the functions which appear most often in its samples and the
code locations which hold it most often. The main thread's time waiting on a lock shows which other thread holds it.
"""

import collections
import gc
import importlib.util
import pathlib
import sys
import threading
import time
import traceback

DEPTH = 6
INTERVAL_S = 0.001


def main() -> None:
    script_path, output_path, label = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve(), sys.argv[3]
    spec = importlib.util.spec_from_file_location("hdf5_replace", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(script_path)]
    parser, arguments = module.parse_arguments()
    hdf5_replace = module.HDF5Replace(arguments, None)
    samples = collections.defaultdict(collections.Counter)  # thread name -> stack text -> count
    tops = collections.defaultdict(collections.Counter)  # thread name -> innermost function -> count
    sample_counts = collections.Counter()
    sampling = threading.Event()
    stop = threading.Event()
    sampler_ident: list[int] = []

    def sample() -> None:
        sampler_ident.append(threading.get_ident())
        names = {}
        while not stop.is_set():
            if sampling.is_set():
                names.update({thread.ident: thread.name for thread in threading.enumerate()})
                for ident, frame in sys._current_frames().items():
                    if ident == sampler_ident[0]:
                        continue
                    name = names.get(ident, str(ident))
                    stack = traceback.extract_stack(frame)[-DEPTH:]
                    sample_counts[name] += 1
                    tops[name][f"{stack[-1].name} ({pathlib.Path(stack[-1].filename).name}:{stack[-1].lineno})"] += 1
                    samples[name][" <- ".join(f"{entry.name} ({pathlib.Path(entry.filename).name}:{entry.lineno})" for entry in reversed(stack))] += 1
            time.sleep(INTERVAL_S)

    sampler = threading.Thread(target=sample, name="sampler", daemon=True)
    sampler.start()
    hdf5_replace.setup()
    try:
        hdf5_replace.open_write_project()
        gc.collect()
        sampling.set()
        times = [hdf5_replace.rewrite() for _ in range(arguments.samples)]
        sampling.clear()
    finally:
        stop.set()
        sampler.join()
        hdf5_replace.teardown()
    lines = [f"{label}: rewrite times ms " + " ".join(f"{t * 1000:.1f}" for t in times)]
    for name, count in sample_counts.most_common():
        lines.append(f"\n== thread {name}: {count} samples")
        lines.append("innermost functions:")
        lines.extend(f"  {n:6d}  {function}" for function, n in tops[name].most_common(12))
        lines.append("stacks:")
        lines.extend(f"  {n:6d}  {stack}" for stack, n in samples[name].most_common(8))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[0])


if __name__ == "__main__":
    main()
