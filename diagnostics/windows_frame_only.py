"""Time only the acquisition frame writes of performance/hdf5_replace.py, with none of its other measurements before them.

Usage, from the nionswift clone, with PYTHONPATH set to the clone, at each commit to compare:
    python windows_frame_only.py <path to hdf5_replace.py> <label> [--rounds N] [--log <file>] [--no-sync] [--no-idle-close] [--pause SECONDS]
        [--no-throttling] [--high-performance]
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

With --pause, it sleeps for the given seconds before each round after the first, as the branch's process mostly sleeps
while the storage settles after a round, to show whether the wait alone slows the next round's frames.

On Windows, --no-throttling opts this process out of power throttling of its execution speed, which ends with the
process, and --high-performance makes the High performance power plan active for the run and makes the plan which was
active before it active again at the end. Each shows whether it keeps the frames after a pause at full speed.

For each block of 128 frames and each round, it also reports the median time of a frame spent inside h5py's dataset
writes, which both versions make for a frame, and the rest of the frame's time apart from it; and the page faults of the
process during the block and its working set and private memory after it, from GetProcessMemoryInfo on Windows, or from
getrusage elsewhere, where the working set is the largest so far and private memory isn't reported.
"""

import argparse
import ctypes
import gc
import importlib.util
import logging
import pathlib
import statistics
import sys
import time
import typing
import uuid

BLOCK_FRAMES = 128


class ProcessMemoryCounters(ctypes.Structure):
    # PROCESS_MEMORY_COUNTERS from psapi.h.
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def get_memory_counters() -> tuple[int, float, float | None]:
    """Return the page faults of this process so far, its working set in MiB, and its private memory in MiB or None."""
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_uint32]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PageFaultCount, counters.WorkingSetSize / 2 ** 20, counters.PagefileUsage / 2 ** 20
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    max_rss = usage.ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10)
    return usage.ru_minflt + usage.ru_majflt, max_rss, None


class Guid(ctypes.Structure):
    # GUID from guiddef.h.
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


class ProcessPowerThrottlingState(ctypes.Structure):
    # PROCESS_POWER_THROTTLING_STATE from processthreadsapi.h.
    _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32), ("StateMask", ctypes.c_uint32)]


# the High performance power plan, SCHEME_MIN.
HIGH_PERFORMANCE_PLAN = uuid.UUID("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")


def opt_out_of_power_throttling() -> None:
    """Opt this process out of power throttling of its execution speed, until it ends."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    # ProcessPowerThrottling is 4 in PROCESS_INFORMATION_CLASS. execution speed (1) in the control mask and not in the
    # state mask turns its throttling off.
    state = ProcessPowerThrottlingState(1, 1, 0)
    if not kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())


def get_active_power_plan() -> uuid.UUID:
    """Return the GUID of the active power plan."""
    powrprof = ctypes.WinDLL("powrprof")
    kernel32 = ctypes.WinDLL("kernel32")
    powrprof.PowerGetActiveScheme.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(Guid))]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    plan = ctypes.POINTER(Guid)()
    result = powrprof.PowerGetActiveScheme(None, ctypes.byref(plan))
    if result != 0:
        raise ctypes.WinError(result)
    try:
        return uuid.UUID(bytes_le=bytes(plan.contents))
    finally:
        kernel32.LocalFree(plan)


def set_active_power_plan(plan: uuid.UUID) -> None:
    """Make the power plan with the given GUID active."""
    powrprof = ctypes.WinDLL("powrprof")
    powrprof.PowerSetActiveScheme.argtypes = [ctypes.c_void_p, ctypes.POINTER(Guid)]
    guid = Guid.from_buffer_copy(plan.bytes_le)
    result = powrprof.PowerSetActiveScheme(None, ctypes.byref(guid))
    if result != 0:
        raise ctypes.WinError(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="time only the frame writes of hdf5_replace.py")
    parser.add_argument("script", help="the path to hdf5_replace.py")
    parser.add_argument("label", help="a label for the results, such as a branch name")
    parser.add_argument("--rounds", type=int, default=3, help="the rounds, each with a new 4D item")
    parser.add_argument("--log", help="the file for the data files log")
    parser.add_argument("--no-sync", action="store_true", help="make each sync of a data file do nothing")
    parser.add_argument("--no-idle-close", action="store_true", help="keep idle data files open")
    parser.add_argument("--pause", type=float, default=0.0, help="the seconds to sleep before each round after the first")
    parser.add_argument("--no-throttling", action="store_true", help="opt this process out of power throttling, on Windows")
    parser.add_argument("--high-performance", action="store_true", help="make the High performance power plan active for the run, on Windows")
    options = parser.parse_args()
    if (options.no_throttling or options.high_performance) and sys.platform != "win32":
        sys.exit("--no-throttling and --high-performance work only on Windows.")
    if options.no_throttling:
        opt_out_of_power_throttling()
        print(f"{options.label}: power throttling turned off for this process", flush=True)
    previous_plan = get_active_power_plan() if options.high_performance else None
    if previous_plan is not None:
        set_active_power_plan(HIGH_PERFORMANCE_PLAN)
        print(f"{options.label}: power plan {get_active_power_plan()} active for the run, was {previous_plan}", flush=True)
    try:
        run(options)
    finally:
        if previous_plan is not None:
            set_active_power_plan(previous_plan)
            print(f"{options.label}: power plan {get_active_power_plan()} active again", flush=True)


def run(options: argparse.Namespace) -> None:
    """Run the rounds which options give, and print their results."""
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
    # the time inside h5py's dataset writes, which both versions make for each frame: master from the data item, the branch
    # from its data handle. this diagnostic wraps the method to time it.
    import h5py
    dataset_write = h5py.Dataset.__setitem__
    write_seconds = [0.0]

    def timed_dataset_write(self: typing.Any, args: typing.Any, value: typing.Any) -> None:
        write_start = time.perf_counter()
        try:
            dataset_write(self, args, value)
        finally:
            write_seconds[0] += time.perf_counter() - write_start

    setattr(h5py.Dataset, "__setitem__", timed_dataset_write)
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
            if round_index > 0 and options.pause > 0:
                time.sleep(options.pause)
            hdf5_replace.begin_frames()
            gc.collect()
            driver_logger.debug("round %d frames started", round_index + 1)
            times = list[float]()
            write_times = list[float]()
            round_faults, _, _ = get_memory_counters()
            block_faults = round_faults
            for frame_index in range(arguments.scan_size ** 2):
                write_seconds[0] = 0.0
                times.append(hdf5_replace.write_frame(frame_index))
                write_times.append(write_seconds[0])
                if len(times) % BLOCK_FRAMES == 0:
                    faults, working_set, private = get_memory_counters()
                    driver_logger.debug("round %d frames %d to %d: median %.3f ms, writes %.3f ms, rest %.3f ms, page faults %d, working set %.0f MiB, private %s",
                                        round_index + 1, len(times) - BLOCK_FRAMES + 1, len(times), statistics.median(times[-BLOCK_FRAMES:]) * 1000,
                                        statistics.median(write_times[-BLOCK_FRAMES:]) * 1000,
                                        statistics.median(t - w for t, w in zip(times[-BLOCK_FRAMES:], write_times[-BLOCK_FRAMES:])) * 1000,
                                        faults - block_faults, working_set, f"{private:.0f} MiB" if private is not None else "not reported")
                    block_faults = faults
            faults, working_set, private = get_memory_counters()
            rest_times = [t - w for t, w in zip(times, write_times)]
            driver_logger.debug("round %d frames finished", round_index + 1)
            frames_written = hdf5_replace.end_frames()
            driver_logger.debug("round %d ended", round_index + 1)
            times.sort()
            print(f"{options.label}  round {round_index + 1}  frame_write  median {statistics.median(times) * 1000:.3f} ms  mean {statistics.mean(times) * 1000:.3f} ms"
                  f"  p90 {times[int(len(times) * 0.9)] * 1000:.3f} ms  writes {statistics.median(write_times) * 1000:.3f} ms  rest {statistics.median(rest_times) * 1000:.3f} ms"
                  f"  page faults per frame {(faults - round_faults) / len(times):.1f}  working set {working_set:.0f} MiB"
                  f"  private {f'{private:.0f} MiB' if private is not None else 'not reported'}  ({len(times)} frames, {frames_written} written)", flush=True)
    finally:
        hdf5_replace.teardown()


if __name__ == "__main__":
    main()
