# Time replacing and writing HDF5 data, and loading a project of HDF5 data, with the project in files.
#
# The project stores all of its data in HDF5 files, in a temporary directory. Each sample times the call on the main
# thread, the document controller's periodic once, and then a pass of the event loop every half second until the
# storage has settled, which is when the data folder holds the same hidden files as before the call. A storage which
# defers the replacement of a data file holds the new data in a hidden file until then, so that a sample includes the
# deferred work which follows on the main thread, but not the waits between the passes. The passes run only the event
# loop, where the document model runs its deferred work, since the rest of the periodic is the window's own work, which
# runs whatever the storage does and would add its time to every pass. The waits keep the processor busy rather than
# sleeping. On a workstation in Windows' Balanced power mode, the work which follows a few seconds of sleep ran about 1.6
# times slower, and a storage which defers its replacements waits seconds after a sample where one which doesn't waits
# not at all, so sleeping would time the two at different speeds.
#
# reshape sets data of a new shape and the same size on an image, alternating between two shapes. reshape_held does the
# same while the image's data is held across the call, as a display holds it, and releases it before the storage
# settles. rewrite sets data of the image's shape and dtype with different values, as a computation does when it
# recomputes its result at a steady shape, and its samples run right after the image is stored, before any reshape.
# rewrite_after_reshape stands for the first recompute after a shape change: each sample reshapes the image and lets the
# storage settle, both untimed, and then times one rewrite, the first after the settled reshape. frame_write reserves a
# 4D item and writes one frame of it in each sample, as an acquisition does: the item is in a transaction and live, and
# each frame is queued as a partial update, which the periodic applies. The storage settles untimed when the acquisition
# ends. reserve starts an acquisition again into the 4D item, which holds data: each sample reserves data of the item's
# shape and dtype, writes a burst of frames as frame_write does, and ends the acquisition, and it times all of these and
# the settling of the storage which follows. load_read opens a project of many images in a new document model and reads
# each image's data once. It has no document controller, so a pass of the event loop takes the place of the periodic.
#
# A measurement's line in the report also gives the mean, which shows a measurement whose samples take two or more
# distinct times better than the median does. The table rows leave it out, so that they match the other scripts' rows.
#
# The details line gives the longest wait of a sample of reshape, reshape_held, rewrite, rewrite_after_reshape and
# reserve for the storage to settle, the rewrites of both rewrite measurements whose values the image's file holds
# afterwards, the frames which hold the written frame once the acquisition ends and after the last reserve, the images
# and the data which the last load read, and, on macOS and Linux, the HDF5 files which this process has open at the end
# of the last load.
#
# The standard case is 10 rewrites of an 8192x8192 float32 image (256 MiB), 10 reshapes of it with and without a holder,
# 10 rewrites of it after a further reshape each, a 32x32 scan of 256x256 frames (256 MiB), 10 reserves of the scan with
# a burst of 16 frames each, and 5 loads of a project of 100 images of 512x512 (1 MiB each), using under 1 GB of disk:
#     python hdf5_replace.py
import argparse
import asyncio
import contextlib
import copy
import cProfile
import datetime
import functools
import gc
import importlib.metadata
import os
import pathlib
import platform
import pstats
import statistics
import sys
import tempfile
import time
import typing

import numpy
import numpy.typing

from nion.data import DataAndMetadata
from nion.swift.model import DataItem
from nion.swift.test import Storage_test
from nion.swift.test import TestContext

# fcntl is only on Unix, and only macOS reads the path of an open file through it.
if sys.platform == "darwin":
    import fcntl

NAME = "hdf5_replace"

# the revision of the measured work.
REVISION = 1

# the packages whose versions the report lists.
PACKAGES = ("nionswift", "nionui", "niondata", "nionutils")

# the seconds between the passes of the event loop while the storage settles.
SETTLE_INTERVAL = 0.5


def positive_int(text: str) -> int:
    """Return text as an integer, for an option which must be at least one."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not at least 1")
    return value


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options which each script shares to parser."""
    parser.add_argument("--label", default="", help="a label for the results, such as a branch name")
    parser.add_argument("--profile", action="store_true", help="print the functions taking the most time, which inflates the times")


def get_statistics(times: typing.Sequence[float]) -> tuple[float, float, float]:
    """Return the median, 90th percentile and interquartile range of times, in milliseconds."""
    times_ms = sorted(time_s * 1000 for time_s in times)
    quartiles = statistics.quantiles(times_ms, n=4)
    iqr_ms = quartiles[2] - quartiles[0]
    return statistics.median(times_ms), times_ms[int(len(times_ms) * 0.9)], iqr_ms


def get_version(distribution_name: str) -> str:
    """Return the installed version of a distribution."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        # a checkout on the path which is not installed has no version.
        return "not installed"


def print_report(parser: argparse.ArgumentParser, arguments: argparse.Namespace, measurements: typing.Sequence[tuple[str, typing.Sequence[float]]],
                 details: typing.Mapping[str, int | float | str]) -> None:
    """Print the report of a run, then one table row for each measurement unless the run was profiled.

    measurements holds the name of each measurement and its times in seconds. details holds counts which confirm that the
    run measured what it intends.
    """
    date_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    options_text = " ".join(f"{name}={value}" for name, value in vars(arguments).items() if name not in ("label", "profile") and value != parser.get_default(name)) or "defaults"
    versions = {distribution_name: get_version(distribution_name) for distribution_name in PACKAGES}
    print(f"{NAME}, revision {REVISION}")
    print(f"date      {date_text}")
    if arguments.label:
        print(f"label     {arguments.label}")
    print(f"machine   {platform.node()}, {platform.platform()}, {os.cpu_count()} processors")
    print(f"software  Python {platform.python_version()}, numpy {numpy.__version__}, " + ", ".join(f"{name} {version}" for name, version in versions.items()))
    print(f"options   {options_text}")
    if details:
        print("details   " + ", ".join(f"{name} {value}" for name, value in details.items()))
    rows = list[str]()
    name_width = max(len(measurement_name) for measurement_name, _ in measurements)
    for measurement_name, times in measurements:
        median_ms, p90_ms, iqr_ms = get_statistics(times)
        mean_ms = statistics.mean(times) * 1000
        print(f"{measurement_name:{name_width}} median {median_ms:.3f} ms  mean {mean_ms:.3f} ms  p90 {p90_ms:.3f} ms  iqr {iqr_ms:.3f} ms  ({len(times)} samples)")
        rows.append(f"| {date_text} | {platform.node()} | nionswift {versions['nionswift']} | {arguments.label} | {options_text} | {measurement_name} | {len(times)} "
                    f"| {median_ms:.3f} | {p90_ms:.3f} | {iqr_ms:.3f} |")
    if arguments.profile:
        print("The profiler inflates the times, so there are no table rows.")
    else:
        print()
        print("\n".join(rows))


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Return the parser and the options from the command line."""
    parser = argparse.ArgumentParser(description="Time replacing and writing HDF5 data, and loading a project of HDF5 data.")
    add_common_arguments(parser)
    parser.add_argument("--samples", type=positive_int, default=10, help="the reshapes timed with and without a holder, the rewrites timed before them and after them, and the reserves timed")
    parser.add_argument("--image-size", type=positive_int, default=8192, help="the width and height of the reshaped float32 image, which must be even")
    parser.add_argument("--scan-size", type=positive_int, default=32, help="the width and height of the scan, one frame for each point")
    parser.add_argument("--frame-size", type=positive_int, default=256, help="the width and height of each float32 frame")
    parser.add_argument("--burst-frames", type=positive_int, default=16, help="the frames written after each reserve, at most the frames of the scan")
    parser.add_argument("--images", type=positive_int, default=100, help="the images in the loaded project")
    parser.add_argument("--load-image-size", type=positive_int, default=512, help="the width and height of each float32 image in the loaded project")
    parser.add_argument("--loads", type=positive_int, default=5, help="the loads timed")
    parser.add_argument("--timeout", type=float, default=60.0, help="the seconds which the storage may take to settle after one sample")
    arguments = parser.parse_args()
    if arguments.image_size % 2:
        parser.error("--image-size must be even")
    if arguments.burst_frames > arguments.scan_size ** 2:
        parser.error("--burst-frames must not be more than the frames of the scan")
    return parser, arguments


def get_time(work: typing.Callable[[], None]) -> float:
    """Return the time in seconds which work takes."""
    start = time.perf_counter()
    work()
    return time.perf_counter() - start


def wait_busy(seconds: float) -> None:
    """Wait for seconds without sleeping, so that the processor stays at the speed of the timed work."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def get_hidden_files(directory: pathlib.Path) -> frozenset[pathlib.Path]:
    """Return the HDF5 files in directory and its folders whose names begin with "."."""
    return frozenset(file_path for file_path in directory.rglob(".*.h5") if file_path.is_file())


def get_open_file_count(directory: pathlib.Path) -> int | None:
    """Return how many HDF5 files in directory and its folders this process has open, or None where it can't be read."""
    if sys.platform == "darwin":
        fd_directory = pathlib.Path("/dev/fd")
    elif sys.platform == "linux":
        fd_directory = pathlib.Path("/proc/self/fd")
    else:
        return None
    resolved_directory = directory.resolve()
    open_count = 0
    for fd_name in os.listdir(fd_directory):
        try:
            if sys.platform == "darwin":
                file_path = pathlib.Path(fcntl.fcntl(int(fd_name), fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode())
            else:
                file_path = pathlib.Path(os.readlink(fd_directory / fd_name))
        except OSError:
            # the descriptor which lists the folder is closed by now, and others may close meanwhile.
            continue
        if file_path.suffix == ".h5" and file_path.is_relative_to(resolved_directory):
            open_count += 1
    return open_count


class HDF5Replace:
    """A project which reshapes and rewrites an image and acquires into a 4D item, and then a project which is loaded and read."""

    def __init__(self, arguments: argparse.Namespace, profiler: cProfile.Profile | None) -> None:
        self.__arguments = arguments
        self.__profiler = profiler
        self.__test_setup: TestContext.TestSetup | None = None
        self.__exit_stack = contextlib.ExitStack()
        self.__project_exit_stack = contextlib.ExitStack()
        self.__random_generator = numpy.random.default_rng(0)

        # the longest a sample of each measurement waited for the storage to settle, in seconds.
        self.longest_settle_times = dict[str, float]()

        # the rewrites whose values the image's file holds afterwards, and the frames which hold the written frame after
        # the last reserve.
        self.rewrites_stored = 0
        self.reserve_frames_written = 0

        # the images and the bytes of data which the last load read, and the HDF5 files open at its end.
        self.load_image_count = 0
        self.load_byte_count = 0
        self.load_open_file_count: int | None = None

    def setup(self) -> None:
        self.__test_setup = TestContext.TestSetup()
        self.__event_loop = asyncio.get_event_loop()
        # the temporary profile context stores its files in the current directory, in every nionswift this measures.
        temporary_directory = self.__exit_stack.enter_context(tempfile.TemporaryDirectory())
        self.__exit_stack.enter_context(contextlib.chdir(pathlib.Path(temporary_directory)))
        self.__exit_stack.callback(self.__project_exit_stack.close)

    def __open_project(self) -> TestContext.FileProfileContext:
        """Return a new profile context for a project which stores all of its data in HDF5."""
        self.__project_exit_stack.close()
        profile_context = self.__project_exit_stack.enter_context(Storage_test.create_temp_profile_context(large_format_size=0))
        self.__projects_directory = profile_context.projects_dir
        return profile_context

    def __time(self, work: typing.Callable[[], None]) -> float:
        """Return the time in seconds which work takes, profiling it when the run is profiled."""
        if self.__profiler:
            self.__profiler.enable()
        elapsed = get_time(work)
        if self.__profiler:
            self.__profiler.disable()
        return elapsed

    def __settle(self, step: typing.Callable[[], None], hidden_files: frozenset[pathlib.Path], *, is_timed: bool = True) -> float:
        """Return the time in seconds which step and the passes of the event loop take, until the data folder holds hidden_files.

        Untimed work isn't profiled either. Raises TimeoutError if the storage doesn't settle within --timeout seconds.
        """
        time_work = self.__time if is_timed else get_time
        deadline = time.monotonic() + self.__arguments.timeout
        elapsed = time_work(step)
        while get_hidden_files(self.__projects_directory) != hidden_files:
            if time.monotonic() > deadline:
                raise TimeoutError(f"the storage didn't settle within {self.__arguments.timeout} seconds")
            wait_busy(SETTLE_INTERVAL)
            elapsed += time_work(self.__run_event_loop)
        return elapsed

    def __settle_sample(self, measurement_name: str, hidden_files: frozenset[pathlib.Path]) -> float:
        """Return the time in seconds which the periodic and the passes of the event loop take, until the data folder holds hidden_files.

        Records how long the storage took to settle, for the details of measurement_name.
        """
        settle_start = time.monotonic()
        settle_time = self.__settle(self.__document_controller.periodic, hidden_files)
        self.longest_settle_times[measurement_name] = max(self.longest_settle_times.get(measurement_name, 0.0), time.monotonic() - settle_start)
        return settle_time

    def open_write_project(self) -> None:
        """Open a project with a document controller, and store the square image in it."""
        profile_context = self.__open_project()
        self.__document_controller = profile_context.create_document_controller()
        image_size = self.__arguments.image_size
        self.__image = self.__random_generator.standard_normal((image_size, image_size), dtype=numpy.float32)
        self.__image_data_item = DataItem.DataItem(self.__image)
        self.__document_controller.document_model.append_data_item(self.__image_data_item)
        self.__settle(self.__document_controller.periodic, frozenset(), is_timed=False)

    def __get_reshaped_image(self, sample_index: int) -> numpy.typing.NDArray[numpy.float32]:
        """Return the image in the shape of the reshape at sample_index."""
        image_size = self.__arguments.image_size
        # the shapes alternate, starting from the square shape which the image is stored with.
        data_shape = (image_size // 2, image_size * 2) if sample_index % 2 == 0 else (image_size, image_size)
        return self.__image.reshape(data_shape)

    def reshape(self, sample_index: int, is_held: bool) -> float:
        """Return the time in seconds to set the image's data to its other shape, and for the storage to settle.

        When is_held, the image's data is held across the call, and released before the storage settles.
        """
        data = self.__get_reshaped_image(sample_index)
        data_shape = data.shape
        data_item = self.__image_data_item
        hidden_files = get_hidden_files(self.__projects_directory)
        if is_held:
            # the holder takes the stored data before the call, as a display does, and lets go of it after.
            with data_item.data_ref() as data_ref:
                held_data = data_ref.data
                call_time = self.__time(functools.partial(data_item.set_data, data))
                del held_data
        else:
            call_time = self.__time(functools.partial(data_item.set_data, data))
        settle_time = self.__settle_sample("reshape_held" if is_held else "reshape", hidden_files)
        assert data_item.data_shape == data_shape, f"the image has shape {data_item.data_shape} rather than {data_shape}"
        return call_time + settle_time

    def rewrite(self, measurement_name: str) -> float:
        """Return the time in seconds to set the image's data to different values of its shape and dtype, and for the storage to settle.

        Records how long the storage took to settle, for the details of measurement_name.
        """
        data_item = self.__image_data_item
        assert data_item.data_shape
        # the values change sign in each sample, in a new array, since the item may keep the array it was given.
        self.__image = numpy.negative(self.__image)
        data = self.__image.reshape(data_item.data_shape)
        hidden_files = get_hidden_files(self.__projects_directory)
        call_time = self.__time(functools.partial(data_item.set_data, data))
        settle_time = self.__settle_sample(measurement_name, hidden_files)
        with data_item.data_ref() as data_ref:
            stored_data = data_ref.data
            assert stored_data is not None
            if numpy.array_equal(numpy.asarray(stored_data[0]), data[0]):
                self.rewrites_stored += 1
        return call_time + settle_time

    def rewrite_after_reshape(self, sample_index: int) -> float:
        """Set the image's data to its shape at sample_index and let the storage settle, both untimed, then return the time in seconds of one rewrite."""
        hidden_files = get_hidden_files(self.__projects_directory)
        self.__image_data_item.set_data(self.__get_reshaped_image(sample_index))
        self.__settle(self.__document_controller.periodic, hidden_files, is_timed=False)
        return self.rewrite("rewrite_after_reshape")

    def __begin_acquisition(self) -> None:
        """Reserve the 4D item's data and begin its acquisition, as an acquisition's data channel does."""
        document_model = self.__document_controller.document_model
        data_item = self.__frame_data_item
        scan_size = self.__arguments.scan_size
        frame_size = self.__arguments.frame_size
        data_item.reserve_data(data_shape=(scan_size, scan_size, frame_size, frame_size), data_dtype=numpy.dtype(numpy.float32), data_descriptor=self.__frame_data_descriptor)
        data_item.increment_data_ref_count()
        self.__frame_transaction = document_model.item_transaction(data_item)
        document_model.begin_data_item_live(data_item)
        data_metadata = data_item.data_metadata
        assert data_metadata
        self.__frame_data_metadata = copy.deepcopy(data_metadata)

    def __end_acquisition(self) -> None:
        """End the 4D item's acquisition, as an acquisition's data channel does."""
        self.__frame_transaction.close()
        self.__document_controller.document_model.end_data_item_live(self.__frame_data_item)
        self.__frame_data_item.decrement_data_ref_count()

    def __count_written_frames(self) -> int:
        """Return the frames of the 4D item which hold the written frame."""
        with self.__frame_data_item.data_ref() as data_ref:
            data = data_ref.data
            assert data is not None
            return int(numpy.count_nonzero(numpy.asarray(data[..., 0, 0]) == self.__frame.data[0, 0, 0, 0]))

    def begin_frames(self) -> None:
        """Make the 4D item and begin its acquisition."""
        frame_size = self.__arguments.frame_size
        self.__frame_data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 2)
        self.__frame_data_item = DataItem.DataItem(large_format=True)
        self.__document_controller.document_model.append_data_item(self.__frame_data_item)
        frame = self.__random_generator.standard_normal((1, 1, frame_size, frame_size), dtype=numpy.float32)
        self.__frame = DataAndMetadata.new_data_and_metadata(frame, data_descriptor=self.__frame_data_descriptor)
        self.__begin_acquisition()
        self.__settle(self.__document_controller.periodic, get_hidden_files(self.__projects_directory), is_timed=False)

    def write_frame(self, frame_index: int) -> float:
        """Return the time in seconds to queue the frame at frame_index, apply it, and for the storage to settle."""
        scan_y, scan_x = divmod(frame_index, self.__arguments.scan_size)
        source_slice = (slice(0, 1), slice(0, 1), slice(None), slice(None))
        destination_slice = (slice(scan_y, scan_y + 1), slice(scan_x, scan_x + 1), slice(None), slice(None))
        hidden_files = get_hidden_files(self.__projects_directory)
        call_time = self.__time(functools.partial(self.__document_controller.document_model.update_data_item_partial, self.__frame_data_item,
                                                  self.__frame_data_metadata, self.__frame, source_slice, destination_slice))
        return call_time + self.__settle(self.__document_controller.periodic, hidden_files)

    def end_frames(self) -> int:
        """End the acquisition, wait for the storage to settle, and return the frames which hold the written frame."""
        self.__end_acquisition()
        self.__settle(self.__document_controller.periodic, frozenset(), is_timed=False)
        return self.__count_written_frames()

    def reserve(self) -> float:
        """Return the time in seconds to reserve the 4D item's data again, write a burst of frames, end the acquisition, and for the storage to settle."""
        hidden_files = get_hidden_files(self.__projects_directory)
        elapsed = self.__time(self.__begin_acquisition)
        for frame_index in range(self.__arguments.burst_frames):
            elapsed += self.write_frame(frame_index)
        elapsed += self.__time(self.__end_acquisition) + self.__settle_sample("reserve", hidden_files)
        self.reserve_frames_written = self.__count_written_frames()
        assert self.reserve_frames_written == self.__arguments.burst_frames, f"{self.reserve_frames_written} of {self.__arguments.burst_frames} frames hold the written frame"
        return elapsed

    def open_load_project(self) -> None:
        """Close the project which was written, and make a project of images to load."""
        self.__profile_context = self.__open_project()
        document_model = self.__profile_context.create_document_model(auto_close=False)
        with document_model.ref():
            load_image_size = self.__arguments.load_image_size
            with document_model.transaction_context():
                for _ in range(self.__arguments.images):
                    document_model.append_data_item(DataItem.DataItem(self.__random_generator.standard_normal((load_image_size, load_image_size), dtype=numpy.float32)))
            self.__settle(self.__run_event_loop, frozenset(), is_timed=False)
        self.__run_event_loop()

    def __run_event_loop(self) -> None:
        """Run one pass of the event loop, as the periodic does."""
        self.__event_loop.stop()
        self.__event_loop.run_forever()

    def __load_and_read(self) -> None:
        """Open the project in a new document model, and read each image's data once."""
        document_model = self.__profile_context.create_document_model(auto_close=False)
        document_model.add_ref()
        self.__load_document_model = document_model
        self.load_image_count = len(document_model.data_items)
        self.load_byte_count = 0
        for data_item in document_model.data_items:
            with data_item.data_ref() as data_ref:
                data = data_ref.data
                assert data is not None
                self.load_byte_count += numpy.asarray(data).nbytes

    def load_read(self) -> float:
        """Return the time in seconds to open the project in a new document model, read each image's data once, and for the storage to settle."""
        elapsed = self.__time(self.__load_and_read) + self.__settle(self.__run_event_loop, frozenset())
        self.load_open_file_count = get_open_file_count(self.__projects_directory)
        self.__load_document_model.remove_ref()
        self.__run_event_loop()
        assert self.load_image_count == self.__arguments.images, f"the load found {self.load_image_count} of {self.__arguments.images} images"
        return elapsed

    def teardown(self) -> None:
        self.__exit_stack.close()
        self.__test_setup = None


def main() -> None:
    parser, arguments = parse_arguments()
    profiler = cProfile.Profile() if arguments.profile else None
    hdf5_replace = HDF5Replace(arguments, profiler)
    hdf5_replace.setup()
    try:
        hdf5_replace.open_write_project()
        gc.collect()
        rewrite_times = [hdf5_replace.rewrite("rewrite") for _ in range(arguments.samples)]
        reshape_times = [hdf5_replace.reshape(sample_index, False) for sample_index in range(arguments.samples)]
        reshape_held_times = [hdf5_replace.reshape(arguments.samples + sample_index, True) for sample_index in range(arguments.samples)]
        rewrite_after_reshape_times = [hdf5_replace.rewrite_after_reshape(2 * arguments.samples + sample_index) for sample_index in range(arguments.samples)]
        hdf5_replace.begin_frames()
        gc.collect()
        frame_write_times = [hdf5_replace.write_frame(frame_index) for frame_index in range(arguments.scan_size ** 2)]
        frames_written = hdf5_replace.end_frames()
        assert frames_written == arguments.scan_size ** 2, f"{frames_written} of {arguments.scan_size ** 2} frames hold the written frame"
        reserve_times = [hdf5_replace.reserve() for _ in range(arguments.samples)]
        hdf5_replace.open_load_project()
        gc.collect()
        load_read_times = [hdf5_replace.load_read() for _ in range(arguments.loads)]
        if profiler:
            pstats.Stats(profiler).sort_stats("tottime").print_stats(15)
    finally:
        hdf5_replace.teardown()
    details: dict[str, int | float | str] = {"image_mib": round(arguments.image_size ** 2 * 4 / 2 ** 20, 3)}
    details.update({f"{measurement_name}_settle_seconds": round(settle_time, 1) for measurement_name, settle_time in hdf5_replace.longest_settle_times.items()})
    details.update({"rewrites_stored": hdf5_replace.rewrites_stored, "frames_written": frames_written, "frame_kib": round(arguments.frame_size ** 2 * 4 / 2 ** 10, 3),
                    "reserve_frames_written": hdf5_replace.reserve_frames_written, "load_images": hdf5_replace.load_image_count,
                    "load_mib": round(hdf5_replace.load_byte_count / 2 ** 20, 3)})
    if hdf5_replace.load_open_file_count is not None:
        details["open_files"] = hdf5_replace.load_open_file_count
    measurements = [("reshape", reshape_times), ("reshape_held", reshape_held_times), ("rewrite", rewrite_times), ("rewrite_after_reshape", rewrite_after_reshape_times),
                    ("frame_write", frame_write_times), ("reserve", reserve_times), ("load_read", load_read_times)]
    print_report(parser, arguments, measurements, details)


if __name__ == "__main__":
    main()
