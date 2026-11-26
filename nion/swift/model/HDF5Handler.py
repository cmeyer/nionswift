"""
    A module for handle .h5 files for Swift.
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import errno
import io
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import threading
import time
import traceback
import typing
import uuid
import weakref

if sys.platform == "darwin":
    import fcntl

import h5py
import numpy
import numpy.typing

from nion.data import DataAndMetadata
from nion.swift.model import DataHandle
from nion.swift.model import StorageHandler
from nion.swift.model import Utility
from nion.utils import DateTime
from nion.utils import Geometry
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.data import DataAndMetadata


PersistentDictType = typing.Dict[str, typing.Any]
_NDArray = numpy.typing.NDArray[typing.Any]

# the log of data file management, which FileStorageSystem describes.
data_files_logger = logging.getLogger("nion.swift.data_files")


def make_directory_if_needed(directory_path: str) -> None:
    """
        Make the directory path, if needed.
    """
    if os.path.exists(directory_path):
        if not os.path.isdir(directory_path):
            raise OSError("Path is not a directory:", directory_path)
    else:
        os.makedirs(directory_path)


def get_write_chunk_shape_for_data(data_shape: DataAndMetadata.ShapeType, data_dtype: numpy.typing.DTypeLike, data_descriptor: DataAndMetadata.DataDescriptor, target_chunk_size: int=240000) -> typing.Optional[DataAndMetadata.ShapeType]:
    """
    Calculate an appropriate write chunk shape for a given data shape and dtype.

    The default target chunk size is 240 kB which seems to be a sweet spot according to benchmarks.
    The algorithm assumes that the data is c-contiguous in memory.

    If the total number of chunks that the calculated chunk shape would lead to is less than 100 (i.e. the file will
    be less than 24 MB in size) or if the data shape is not suitable for chunking, return None.

    The chunk shape is calculated according to the following rules:
    - Use a quarter of the data axis (i.e. the last axis for 1D data and the last two axis for 2D data)
      if the number of elements in the last two or last three dimensions is large enough. If not, use
      the entire data axis.
    - Work backwards from the last axis and increse the number of elements per axis until prod(chunk_chape) >= target_chunk size.
      A chunk cannot exceed the number of elements of an axis, so if one dimension is "full", move to the next axis.
    - The slow scan axis and the sequence axis when acquiring a sequence of SIs or 4D STEM images can never be chunked (i.e.
      chunk shape == 1 for these axis).
    - The sequence axis in a sequence of 1D or 2D data can be chunked.
    """
    data_dtype = numpy.dtype(data_dtype)

    is_2d_data = data_descriptor.datum_dimension_count == 2

    target_chunk_size = int(round(target_chunk_size / data_dtype.itemsize))
    divisors = [1] * len(data_shape)
    if len(data_shape) > 2 and (
        (numpy.prod(data_shape[-3:]) > 8 * target_chunk_size and is_2d_data) or
        (numpy.prod(data_shape[-2:]) > 2 * target_chunk_size)):
        if is_2d_data:
            divisors[-2:] = [4, 4]
        else:
            divisors[-1] = 4

    chunk_size = 1
    counter = len(data_shape)
    chunk_shape = [1] * len(data_shape)
    while chunk_size < target_chunk_size and counter > 0:
        counter -= 1
        chunk_size *= data_shape[counter] // divisors[counter]
        chunk_shape[counter] = data_shape[counter] // divisors[counter]

    if chunk_size == 0: # This means one of the input dimensions was "0", so chunking cannot be used
        return None

    chunk_size //= chunk_shape[counter]
    remaining_elements = min(max(target_chunk_size // chunk_size, 1), data_shape[counter])
    chunk_shape[counter] = int(remaining_elements)

    n_chunks = 1
    for i in range(len(chunk_shape)):
        n_chunks *= data_shape[i] // chunk_shape[i]
    if n_chunks < 100:
        return None

    # Do not allow chunking of the slow scan axis or the sequence axis when acquiring a sequence of SIs or 4D STEM images.
    if is_2d_data and len(data_shape) > 3:
        chunk_shape[:-3] = [1] * (len(data_shape) - 3)
    elif not is_2d_data and len(data_shape) > 2:
        chunk_shape[:-2] = [1] * (len(data_shape) - 2)

    return tuple(chunk_shape)


_HDF5FilePointer = typing.Any

# the version written into the properties of a removed data item's file while the file stays at its own path. earlier
# versions and this one skip a data file whose version is not the one they write.
REMOVED_FILE_VERSION = -1

# the name of a pending file, which holds a data item's new data beside its canonical file <stem>.h5 until it replaces
# it, as find_pending_file_path makes it for a canonical file of any name. earlier versions and this one skip names
# starting with a dot when they look for data files.
_PENDING_FILE_NAME_PATTERN = re.compile(r"\.(.+)\.(\d+)\.h5")

# the name of a ready file, a synced pending file whose replace was still refused when its project closed, as
# get_ready_file_path makes it. it differs from every pending file name, which ends in a number.
_READY_FILE_NAME_PATTERN = re.compile(r"\.(.+)\.ready\.h5")

# the attribute of a ready file's dataset which holds, from get_stored_modified_json, the stored modified time of its
# canonical file when the ready file was made.
_READY_CANONICAL_MODIFIED_ATTRIBUTE = "ready_canonical_modified"

# the time in seconds between the tries of refused replaces when a project closes.
_CLOSE_REPLACE_RETRY_INTERVAL = 0.05

# the time in seconds for which a project close tries refused renames to ready files again, after its tries of the replaces,
# since a virus scanner may open a pending file as its last handle closes.
_CLOSE_READY_RENAME_TIMEOUT = 1.0


# the key of a path in the open paths. this process makes the path of each data file from its project's data folder, so
# the same file always has the same path, up to its case and separators on Windows, and the key doesn't resolve symlinks,
# which on Windows opens the file.
def _get_open_path_key(path: pathlib.Path) -> str:
    return os.path.normcase(os.path.abspath(path))


# make the contents of the file at file_path durable on disk. it opens a handle of its own for writing, which Windows
# needs to sync a file. a sync makes the file's written data durable whichever handle wrote it, so this works while h5py
# has the file open.
def _sync_file(file_path: pathlib.Path) -> None:
    with open(file_path, "r+b") as file:
        # macOS fsync doesn't flush the drive's cache, and F_FULLFSYNC does; a file system without it gets fsync.
        if sys.platform == "darwin" and hasattr(fcntl, "F_FULLFSYNC"):
            try:
                fcntl.fcntl(file.fileno(), fcntl.F_FULLFSYNC)
                return
            except OSError:
                pass
        os.fsync(file.fileno())


def move_data_file(file_path: pathlib.Path, new_file_path: pathlib.Path | None, *, replace: bool = False, reason: str | None = None) -> None:
    """Move the data file at file_path to new_file_path, or delete it when new_file_path is None.

    With replace, the file atomically replaces any file at new_file_path, which must be on the same file system. The reason,
    if given, goes into the data files log with the move.

    Raises PermissionError, changing nothing, if either path is open in this process, as Windows refuses to move, replace
    or delete a file this process has open. Raises OSError if the file system refuses.
    """
    for path in (file_path, new_file_path):
        if path is not None and HDF5FileEntry.is_path_open(path):
            raise PermissionError(errno.EACCES, "Data file is open in this process", str(path))
    if new_file_path is None:
        os.remove(file_path)
        if reason is not None:
            data_files_logger.debug("data file deleted: %s, reason %s", file_path, reason)
        else:
            data_files_logger.debug("data file deleted: %s", file_path)
    elif replace:
        os.replace(file_path, new_file_path)
        if reason is not None:
            data_files_logger.debug("data file replaced: %s, replaced by %s, reason %s", new_file_path, file_path, reason)
        else:
            data_files_logger.debug("data file replaced: %s, replaced by %s", new_file_path, file_path)
    else:
        try:
            # shutil.move copies the file when the rename fails for any reason, so a refused rename would cost a whole copy.
            os.rename(file_path, new_file_path)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            # another file system mounted on the data folder's trash or date folders.
            shutil.move(file_path, new_file_path)
        if reason is not None:
            data_files_logger.debug("data file moved: %s, to %s, reason %s", file_path, new_file_path, reason)
        else:
            data_files_logger.debug("data file moved: %s, to %s", file_path, new_file_path)


def move_data_file_to_trash(file_path: pathlib.Path, trash_file_path: pathlib.Path, *, reason: str | None = None) -> None:
    """Move the data file at file_path to trash_file_path, making the trash folder if needed.

    A file of the same name already in the trash stays, and the data file is deleted instead. The reason, if given, goes
    into the data files log with the move or delete. Raises as move_data_file.
    """
    if trash_file_path.exists():
        move_data_file(file_path, None, reason=reason)
    else:
        trash_file_path.parent.mkdir(exist_ok=True)
        move_data_file(file_path, trash_file_path, reason=reason)


def find_pending_file_path(file_path: pathlib.Path, index: int) -> tuple[pathlib.Path, int]:
    """Return the first pending file name beside the canonical file at file_path, numbered from index, which is free.

    Returns the path and its number.
    """
    while (pending_file_path := file_path.with_name(f".{file_path.stem}.{index}{file_path.suffix}")).exists():
        index += 1
    return pending_file_path, index


def get_ready_file_path(file_path: pathlib.Path) -> pathlib.Path:
    """Return the path of the ready file beside the canonical file at file_path."""
    return file_path.with_name(f".{file_path.stem}.ready{file_path.suffix}")


def get_canonical_file_path(pending_file_path: pathlib.Path) -> pathlib.Path | None:
    """Return the path of the canonical file which the pending file at pending_file_path replaces, or None if it is not a pending file."""
    if match := _PENDING_FILE_NAME_PATTERN.fullmatch(pending_file_path.name):
        return pending_file_path.with_name(match.group(1) + ".h5")
    return None


def get_ready_canonical_file_path(ready_file_path: pathlib.Path) -> pathlib.Path | None:
    """Return the path of the canonical file which the ready file at ready_file_path replaces, or None if it is not a ready file."""
    if match := _READY_FILE_NAME_PATTERN.fullmatch(ready_file_path.name):
        return ready_file_path.with_name(match.group(1) + ".h5")
    return None


def get_stored_modified_json(properties: typing.Mapping[str, typing.Any]) -> str:
    """Return the stored modified time in the properties as json, in which a missing time has a value of its own."""
    return json.dumps(properties.get("modified"))


def _parse_properties(json_properties: typing.Any) -> PersistentDictType | None:
    # return the properties in a dataset's properties attribute, which h5py doesn't type, or None if it doesn't hold any.
    try:
        properties = json.loads(json_properties)
    except (TypeError, ValueError):
        # an attribute which isn't json holds no properties to read.
        return None
    return properties if isinstance(properties, dict) else None


def _read_file_properties(file_path: pathlib.Path, *, file_locking: bool) -> PersistentDictType | None:
    # return the properties in the HDF5 file at file_path, opening it read only, or None if they can't be read.
    try:
        with h5py.File(file_path, "r", locking=None if file_locking else False) as fp:
            return _parse_properties(fp["data"].attrs["properties"])
    except (OSError, KeyError):
        # a file which is missing, damaged, or has no properties has none to read.
        return None


def _read_ready_canonical_modified(ready_file_path: pathlib.Path, *, file_locking: bool) -> str | None:
    # return the canonical file's stored modified time as the ready file recorded it, or None if the ready file has no
    # record or its properties can't be read.
    try:
        with h5py.File(ready_file_path, "r", locking=None if file_locking else False) as fp:
            attributes = fp["data"].attrs
            return str(attributes[_READY_CANONICAL_MODIFIED_ATTRIBUTE]) if _parse_properties(attributes["properties"]) is not None else None
    except (OSError, KeyError):
        # a damaged file, or a file with no properties or no record, has no record to read.
        return None


# mark the dataset's properties removed, leaving the rest of them as they are; return the data item uuid. the dataset is an
# h5py dataset, which has no type information.
def _mark_dataset_removed(dataset: typing.Any) -> uuid.UUID:
    properties = typing.cast(PersistentDictType, json.loads(dataset.attrs.get("properties", "")))
    properties["version"] = REMOVED_FILE_VERSION
    dataset.attrs["properties"] = json.dumps(properties)
    return uuid.UUID(properties["uuid"])


# mark the closed file at file_path removed. raises OSError if it refuses.
def _mark_file_removed(file_path: pathlib.Path, *, file_locking: bool) -> None:
    file = HDF5FileEntry(file_path, locking=file_locking)
    try:
        data_item_uuid = _mark_dataset_removed(file.fp["data"])
        file.fp.flush()
        data_files_logger.debug("data file marked removed: %s, data item %s, file type HDF5", file_path, data_item_uuid)
    finally:
        file.close("marked removed")


class HDF5FileEntry:
    """An HDF5 file at a path, opened when first used and owned by one handler.

    Each handler opens its own file, so closing it affects only that handler's data. The file may be opened again after
    it is closed. The count of open files, which get_open_count returns, is kept for the data files log and for detecting
    unclosed files in tests, and the paths of open files so that move_data_file refuses to move or delete a file which
    this process has open.
    """
    _open_count = 0
    __open_count_lock = threading.Lock()
    # the paths of the open files, each with the number of entries which have it open, so that nothing in this process
    # moves or deletes a file it has open. the open count lock protects it.
    __open_paths: dict[str, int] = collections.defaultdict(int)

    def __init__(self, path: pathlib.Path, *, locking: bool = True) -> None:
        self.__lock = threading.RLock()
        self.__path = path
        # whether HDF5 locks the file while it is open.
        self.__locking = locking
        self.__fp: typing.Optional[_HDF5FilePointer] = None
        # whether this file is queued as idle, waiting to be taken in by the idle files. it is set when the file is
        # queued, under its handler's lock or in a finalizer, and cleared on the main thread as the file is taken in. it
        # needs no lock of its own: reading or writing it is atomic, and it is set before the file is queued and cleared
        # only after it is taken, so a file which is not queued again while it is set is never lost. two threads
        # queueing the file at once may both queue it, which is harmless.
        self.is_queued_idle = False

    @property
    def fp(self) -> _HDF5FilePointer:
        with self.__lock:
            self.open()
            assert self.__fp
            return self.__fp

    @property
    def path(self) -> pathlib.Path:
        return self.__path

    def open(self) -> None:
        with self.__lock:
            if not self.__fp:
                self.__path.parent.mkdir(parents=True, exist_ok=True)
                # None keeps HDF5's own locking.
                self.__fp = h5py.File(self.__path, "a", locking=None if self.__locking else False)
                with HDF5FileEntry.__open_count_lock:
                    HDF5FileEntry._open_count += 1
                    HDF5FileEntry.__open_paths[_get_open_path_key(self.__path)] += 1
                data_files_logger.debug("data file opened: %s, file type HDF5", self.__path)

    def close(self, reason: str | None = None) -> None:
        """Close the file if it is open. The reason, if given, goes into the data files log with the close."""
        with self.__lock:
            if self.__fp:
                self.__fp.close()
                with HDF5FileEntry.__open_count_lock:
                    HDF5FileEntry._open_count -= 1
                    open_path_key = _get_open_path_key(self.__path)
                    HDF5FileEntry.__open_paths[open_path_key] -= 1
                    if HDF5FileEntry.__open_paths[open_path_key] == 0:
                        del HDF5FileEntry.__open_paths[open_path_key]
                if reason is not None:
                    data_files_logger.debug("data file closed: %s, file type HDF5, reason %s", self.__path, reason)
                else:
                    data_files_logger.debug("data file closed: %s, file type HDF5", self.__path)
            self.__fp = None

    @staticmethod
    def get_open_count() -> int:
        """Return the number of HDF5 files open in this process."""
        with HDF5FileEntry.__open_count_lock:
            return HDF5FileEntry._open_count

    @staticmethod
    def is_path_open(path: pathlib.Path) -> bool:
        """Return whether any entry in this process has the file at path open."""
        with HDF5FileEntry.__open_count_lock:
            return _get_open_path_key(path) in HDF5FileEntry.__open_paths

    @property
    def is_open(self) -> bool:
        with self.__lock:
            return self.__fp is not None


@dataclasses.dataclass(frozen=True)
class HDF5RemovedFile:
    """A file of a removed data item, which moves into the trash once this process no longer has it open.

    A visible file is marked removed meanwhile; a pending file is hidden already.
    """

    data_item_uuid: uuid.UUID
    """The uuid of the removed data item, by which a restore finds the file."""

    file_path: pathlib.Path
    """The path of the file, which stays there until it moves."""

    trash_file_path: pathlib.Path
    """The path in the trash to which the file moves, named as the data item's canonical file."""

    canonical_file_path: pathlib.Path | None = None
    """The data item's canonical file when the file is a pending file which replaces it, otherwise None.

    The canonical file stays, marked removed, until the pending file moves into the trash, and is then deleted.
    """

    canonical_modified: str | None = None
    """The stored modified time which the pending file remembers of its canonical file, from get_stored_modified_json."""


class HDF5FileSync:
    """A sync of a file which the deferred files run on their sync thread, done once it finishes or is dropped.

    A sync which the file system refuses, which fails with an unexpected error, which is dropped because its file went,
    or which is requested after its deferred files closed, carries the error.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        self.file_path = file_path
        # the error is set before the event, and read only once the event is set.
        self.error: Exception | None = None
        self.__done_event = threading.Event()

    @property
    def is_done(self) -> bool:
        return self.__done_event.is_set()

    def wait(self) -> None:
        """Wait until the sync is done."""
        self.__done_event.wait()

    def _finish(self, error: Exception | None) -> None:
        self.error = error
        self.__done_event.set()


class HDF5DeferredFiles:
    """The HDF5 files which their handlers have given up while still in use, and the files waiting to be renamed or deleted.

    A retired file stays open so that the holders of its data handle keep reading its data. When the last holder drops
    the handle, a finalizer queues the file, and close_released_files closes it on the main thread. Close closes every
    retired file, whether or not its holders have gone.

    Nothing in this process renames, moves or deletes a file it has open. Once a file is closed, close_released_files and
    close do what is waiting for it:
    - a file to delete, such as a pending file which a later one superseded, is deleted;
    - a removed file moves into the trash. A restore before the move takes the file back, and it is then only closed;
    - a pending file replaces its canonical file once this process has closed both, or once its handler finds them free.

    A pending file is synced on a sync thread before it replaces its canonical file or moves into the trash, so that the
    main thread doesn't wait for the disk. A pass requests the sync of a pending file which its handler still has once
    the file has had no writes for sync_delay, in the units of the clock, which returns the current time.

    An operation which the file system refuses is tried again at the next pass, including the mark of the canonical file
    of a removed pending file. Close tries the refused replacements again for close_replace_timeout seconds in all, and
    renames a pending file whose replacement is still refused to its ready file (see get_ready_file_path), which records
    the canonical file's stored modified time for the next load, trying a refused rename again for a short time of its
    own. Close waits between its tries with close_retry_wait, which is given the time in seconds. The time recorded is
    the one given with the pending file, since another program may refuse reads of the canonical file by then. Close
    leaves anything else still refused for the next load.

    A replacement is stuck once the file system has refused it for stuck_replace_delay, by the clock, since the first
    refusal of its pending file, and take_stuck_replacements returns it once.
    """

    def __init__(self, *, sync_delay: float = 0.0, clock: typing.Callable[[], float] = time.monotonic, close_replace_timeout: float = 0.0,
                 close_retry_wait: typing.Callable[[float], None] = time.sleep, file_locking: bool = True, stuck_replace_delay: float = 0.0) -> None:
        # how long a pending file goes without writes before a pass requests its sync, and the clock which times it.
        self.__sync_delay = sync_delay
        self.__clock = clock
        # how long, by the clock, the file system refuses the replace of a pending file before it is stuck.
        self.__stuck_replace_delay = stuck_replace_delay
        # how long close tries the refused replacements again, in seconds of real time, since close waits between tries,
        # and the function which waits.
        self.__close_replace_timeout = close_replace_timeout
        self.__close_retry_wait = close_retry_wait
        # whether HDF5 locks the files opened here.
        self.__file_locking = file_locking
        # the retired files and the finalizers of their data handles. files are retired on any thread; the lock protects
        # the dict, and the dicts below.
        self.__lock = threading.Lock()
        self.__retired_files: dict[HDF5FileEntry, weakref.finalize[[HDF5FileEntry], DataHandle.DataHandle]] = dict()
        # the retired files whose last holder has gone. a finalizer may run on any thread, inside unrelated code, so it
        # only appends; deque appends and pops are thread safe, and only the main thread pops.
        self.__released_files: collections.deque[HDF5FileEntry] = collections.deque()
        # the removed files not yet moved into the trash, by data item uuid.
        self.__removed_files: dict[uuid.UUID, HDF5RemovedFile] = dict()
        # the files to delete once nothing in this process has them open, by open path key.
        self.__deleted_files: dict[str, pathlib.Path] = dict()
        # the canonical files of removed pending files whose mark was refused, until they are marked, deleted or restored.
        # main thread only.
        self.__unmarked_files: set[pathlib.Path] = set()
        # the pending files, each with its canonical file, a weak reference to the handler whose current file it is, or
        # None once that handler has closed and left the replacement to these files, and the canonical file's stored
        # modified time from get_stored_modified_json, or None if it couldn't be read when the file started pending.
        self.__pending_files: dict[pathlib.Path, tuple[pathlib.Path, weakref.ReferenceType[HDF5Handler] | None, str | None]] = dict()
        # the syncs of removed pending files, which move into the trash once synced, by data item uuid. main thread only.
        self.__removed_file_syncs: dict[uuid.UUID, HDF5FileSync] = dict()
        # the time of the first refused replace of each pending file, by the clock, or None once it is found stuck, so
        # that it is found stuck once; and the replacements found stuck since the last take, each as its pending file and
        # its canonical file.
        self.__refused_replace_times: dict[pathlib.Path, float | None] = dict()
        self.__stuck_replacements: list[tuple[pathlib.Path, pathlib.Path]] = list()

        # the syncs waiting for the sync thread, oldest first and one per file, so that the files bound them; the sync
        # running; and the thread, which the first request starts. the condition protects them, and the thread doesn't
        # hold it during a sync.
        self.__sync_condition = threading.Condition()
        self.__sync_requests: dict[pathlib.Path, HDF5FileSync] = dict()
        self.__running_sync: HDF5FileSync | None = None
        self.__sync_thread: threading.Thread | None = None
        self.__is_sync_closed = False
        # whether a sync was requested since the last pass. a pending file closes for its sync, so the request is a change
        # which the pass reports, whichever caller made it. the condition protects it.
        self.__is_sync_requested = False
        # the files whose last sync failed with an unexpected error, whose traceback has been printed, until they sync or
        # go. the condition protects them.
        self.__unexpected_error_file_paths: set[pathlib.Path] = set()

    def get_time(self) -> float:
        """Return the current time by the clock which times the syncs. Any thread."""
        return self.__clock()

    def is_sync_due(self, last_write_time: float) -> bool:
        """Return whether a file last written at last_write_time, by the clock, has had no writes for the sync delay. Any thread."""
        return self.__clock() - last_write_time >= self.__sync_delay

    @property
    def retired_count(self) -> int:
        """Return the number of retired files not yet closed."""
        with self.__lock:
            return len(self.__retired_files)

    @property
    def pending_count(self) -> int:
        """Return the number of pending files not yet replacing their canonical files."""
        with self.__lock:
            return len(self.__pending_files)

    def retire(self, file: HDF5FileEntry, data_handle: DataHandle.DataHandle) -> None:
        # may be called on any thread.
        with self.__lock:
            self.__retired_files[file] = weakref.finalize(data_handle, self.__released_files.append, file)
        data_files_logger.debug("data file retired while its data is held: %s, file type HDF5", file.path)

    def delete_file(self, file_path: pathlib.Path) -> bool:
        """Delete the file now, or once nothing in this process has it open; return whether it is gone now.

        A file which is missing counts as gone. May be called on any thread.
        """
        if not HDF5FileEntry.is_path_open(file_path):
            try:
                move_data_file(file_path, None)
                return True
            except FileNotFoundError:
                return True
            except OSError as e:
                data_files_logger.debug("data file delete refused, to be tried again: %s, error %s", file_path, e)
        with self.__lock:
            self.__deleted_files[_get_open_path_key(file_path)] = file_path
        return False

    def is_deleted_file(self, file_path: pathlib.Path) -> bool:
        # main thread only. whether the file waits to be deleted.
        with self.__lock:
            return _get_open_path_key(file_path) in self.__deleted_files

    def keep_file(self, file_path: pathlib.Path) -> bool:
        # main thread only. the file waiting to be deleted is kept after all; return whether it was waiting.
        with self.__lock:
            return self.__deleted_files.pop(_get_open_path_key(file_path), None) is not None

    def add_removed_file(self, removed_file: HDF5RemovedFile) -> None:
        # main thread only. the file moves into the trash at the first pass after nothing in this process has it open.
        with self.__lock:
            self.__removed_files[removed_file.data_item_uuid] = removed_file

    def restore_removed_file(self, data_item_uuid: uuid.UUID) -> HDF5RemovedFile | None:
        """Take back the removed file of the data item, which then stays where it is; return it, or None.

        Main thread only.
        """
        self.__removed_file_syncs.pop(data_item_uuid, None)
        with self.__lock:
            return self.__removed_files.pop(data_item_uuid, None)

    def add_unmarked_file(self, file_path: pathlib.Path) -> None:
        # main thread only. the file refused its mark, which is tried again at each pass.
        self.__unmarked_files.add(file_path)

    def forget_unmarked_file(self, file_path: pathlib.Path) -> bool:
        # main thread only. the file no longer needs its mark. return whether it was waiting for one, in which case it
        # has none.
        if file_path in self.__unmarked_files:
            self.__unmarked_files.remove(file_path)
            return True
        return False

    def get_restore_file_paths(self, data_item_uuid: uuid.UUID, canonical_file_name: str) -> typing.Sequence[pathlib.Path]:
        """Return the files which a restore of the data item writes, of those these files know.

        The canonical file of a pending file already in the trash waits to be deleted under canonical_file_name. Main thread only.
        """
        with self.__lock:
            removed_file = self.__removed_files.get(data_item_uuid)
            deleted_file_paths = list(self.__deleted_files.values())
        file_paths = list[pathlib.Path]()
        if removed_file is not None:
            file_paths.append(removed_file.file_path)
            if removed_file.canonical_file_path is not None:
                file_paths.append(removed_file.canonical_file_path)
        else:
            file_paths.extend(file_path for file_path in deleted_file_paths if file_path.name == canonical_file_name)
        return [file_path for file_path in file_paths if file_path not in self.__unmarked_files and file_path.exists()]

    def is_canonical_file_pending(self, canonical_file_path: pathlib.Path) -> bool:
        # main thread only. whether a pending file still waits to replace the canonical file.
        with self.__lock:
            return any(pending_file[0] == canonical_file_path for pending_file in self.__pending_files.values())

    def add_pending_file(self, pending_file_path: pathlib.Path, canonical_file_path: pathlib.Path, handler: HDF5Handler, canonical_modified: str | None) -> None:
        # any thread, under the handler's lock. the handler completes the replacement when it can.
        with self.__lock:
            self.__pending_files[pending_file_path] = (canonical_file_path, weakref.ref(handler), canonical_modified)

    def get_pending_canonical_modified(self, pending_file_path: pathlib.Path) -> str | None:
        # any thread, under the handler's lock. the time given with the pending file.
        with self.__lock:
            pending_file = self.__pending_files.get(pending_file_path)
            return pending_file[2] if pending_file else None

    def leave_pending_file(self, pending_file_path: pathlib.Path) -> None:
        # any thread, under the handler's lock. the handler closes; these files complete the replacement. the handler
        # leaves only a pending file which holds the properties which describe its data and is synced.
        with self.__lock:
            if pending_file := self.__pending_files.get(pending_file_path):
                self.__pending_files[pending_file_path] = (pending_file[0], None, pending_file[2])

    def remove_pending_file(self, pending_file_path: pathlib.Path) -> None:
        # any thread, under the handler's lock. the replacement completed, or the pending file went, and a sync of it
        # still waiting is dropped.
        with self.__lock:
            self.__pending_files.pop(pending_file_path, None)
            self.__refused_replace_times.pop(pending_file_path, None)
        self.drop_sync(pending_file_path)

    def discard_pending_file(self, pending_file_path: pathlib.Path, canonical_file_path: pathlib.Path) -> None:
        # any thread, under the handler's lock. the pending file is deleted once nothing has it open. an earlier session
        # saved the data of a ready file, so it stays, and the next load decides whether it replaces its canonical file.
        self.remove_pending_file(pending_file_path)
        if pending_file_path != get_ready_file_path(canonical_file_path):
            self.delete_file(pending_file_path)

    def record_refused_replace(self, pending_file_path: pathlib.Path, canonical_file_path: pathlib.Path) -> None:
        # any thread, under the handler's lock. the file system refused the replace of the canonical file by the pending
        # file. the replace is stuck once it has been refused for the stuck delay.
        now = self.__clock()
        with self.__lock:
            refused_replace_time = self.__refused_replace_times.setdefault(pending_file_path, now)
            if refused_replace_time is not None and now - refused_replace_time >= self.__stuck_replace_delay:
                data_files_logger.debug("pending replacement stuck: %s, replaces %s", pending_file_path, canonical_file_path)
                self.__stuck_replacements.append((pending_file_path, canonical_file_path))
                self.__refused_replace_times[pending_file_path] = None

    def take_stuck_replacements(self) -> typing.Sequence[tuple[pathlib.Path, pathlib.Path]]:
        """Return the replacements found stuck since the last call, each as its pending file and its canonical file.

        Each replacement is returned once. Main thread only.
        """
        with self.__lock:
            stuck_replacements = self.__stuck_replacements
            self.__stuck_replacements = list()
        return stuck_replacements

    def request_sync(self, file_path: pathlib.Path) -> HDF5FileSync:
        """Return a sync of the file at file_path on the sync thread, which covers what was written to it before this call.

        A sync of the file which is still waiting covers it, and is returned. A sync requested after close is finished
        already, with an error. Any thread.
        """
        with self.__sync_condition:
            if self.__is_sync_closed:
                closed_sync = HDF5FileSync(file_path)
                closed_sync._finish(OSError(f"Data file sync requested after its deferred files closed: {file_path}"))
                return closed_sync
            if (sync := self.__sync_requests.get(file_path)) is None:
                sync = HDF5FileSync(file_path)
                self.__sync_requests[file_path] = sync
                self.__is_sync_requested = True
                if self.__sync_thread is None:
                    self.__sync_thread = threading.Thread(target=self.__sync_files, name="HDF5 file sync", daemon=True)
                    self.__sync_thread.start()
                # the sync thread and _wait_for_syncs both wait on the condition.
                self.__sync_condition.notify_all()
            return sync

    def _wait_for_syncs(self) -> None:
        # wait until the sync thread has run every sync requested so far. for testing.
        with self.__sync_condition:
            self.__sync_condition.wait_for(lambda: not self.__sync_requests and self.__running_sync is None)

    # drop the waiting sync of the file at file_path, which went, if any. any thread.
    def drop_sync(self, file_path: pathlib.Path) -> None:
        with self.__sync_condition:
            sync = self.__sync_requests.pop(file_path, None)
            self.__unexpected_error_file_paths.discard(file_path)
        if sync is not None:
            sync._finish(FileNotFoundError(errno.ENOENT, "Data file went before its sync", str(file_path)))

    # the sync thread. it syncs each file through a handle of its own, and takes no other lock.
    def __sync_files(self) -> None:
        while True:
            with self.__sync_condition:
                while not self.__sync_requests and not self.__is_sync_closed:
                    self.__sync_condition.wait()
                if self.__is_sync_closed:
                    return
                file_path, sync = next(iter(self.__sync_requests.items()))
                del self.__sync_requests[file_path]
                self.__running_sync = sync
            try:
                _sync_file(file_path)
                with self.__sync_condition:
                    self.__unexpected_error_file_paths.discard(file_path)
                sync._finish(None)
            except OSError as e:
                sync._finish(e)
            except Exception as e:
                # the thread goes on to serve the later syncs, and this one finishes with the error, as a refused one.
                # the sync is requested again at each pass, so the traceback is printed once until the file syncs.
                with self.__sync_condition:
                    is_repeated_error = file_path in self.__unexpected_error_file_paths
                    self.__unexpected_error_file_paths.add(file_path)
                if is_repeated_error:
                    data_files_logger.debug("data file sync failed again: %s, error %s", file_path, e)
                else:
                    traceback.print_exc()
                sync._finish(e)
            with self.__sync_condition:
                self.__running_sync = None
                self.__sync_condition.notify_all()

    def close_released_files(self) -> bool:
        # main thread only. close the files whose last holder has gone, do what waits for the closed files and their
        # syncs, then complete the pending replacements. return whether anything changed.
        is_changed = bool(self.__released_files)
        while self.__released_files:
            file = self.__released_files.popleft()
            with self.__lock:
                self.__retired_files.pop(file, None)
            file.close("released by its last holder")
        self.__mark_unmarked_files()
        is_changed = self.__delete_files() or is_changed
        is_changed = self.__move_removed_files() or is_changed
        is_changed = self.__complete_pending_files() or is_changed
        with self.__sync_condition:
            is_changed = self.__is_sync_requested or is_changed
            self.__is_sync_requested = False
        return is_changed

    def close(self) -> None:
        # main thread only, after the handlers have closed. what is still refused stays for the next load: a removed
        # file stays at its path, marked removed, and the load deletes it; a pending file which can't become its ready
        # file stays, and the load deletes it; a file to delete stays, and the load deletes it if it is a pending file or
        # marked removed; a canonical file which still refuses its mark stays, and the load loads it.
        with self.__lock:
            retired_files = self.__retired_files
            self.__retired_files = dict()
        for file, finalizer in retired_files.items():
            finalizer.detach()
            file.close("project closed")
        self.__released_files.clear()
        self.__mark_unmarked_files()
        self.__delete_files()
        self.__move_removed_files()
        self.__complete_pending_files()
        # another process may have a canonical file open for a moment, as a virus scanner does, and Windows refuses the
        # replace meanwhile. once the handlers have closed, the pending files left are those whose replace is refused.
        deadline = time.monotonic() + self.__close_replace_timeout
        while self.__has_pending_files() and time.monotonic() < deadline:
            self.__close_retry_wait(_CLOSE_REPLACE_RETRY_INTERVAL)
            self.__complete_pending_files()
        self.__make_refused_pending_files_ready()
        for file_path in self.__unmarked_files:
            if file_path.exists():
                logging.getLogger("loader").warning(f"Unable to mark data file {file_path} removed. Its removed data item comes back at the next load, with no display item.")
        self.__unmarked_files.clear()
        with self.__lock:
            self.__removed_files.clear()
            self.__deleted_files.clear()
            self.__pending_files.clear()
            self.__refused_replace_times.clear()
            self.__stuck_replacements.clear()
        self.__removed_file_syncs.clear()
        # the handlers have done their syncs. a removed pending file which waits for its sync stays for the next load.
        with self.__sync_condition:
            self.__is_sync_closed = True
            dropped_syncs = list(self.__sync_requests.values())
            self.__sync_requests.clear()
            sync_thread = self.__sync_thread
            self.__sync_condition.notify_all()
        for sync in dropped_syncs:
            sync._finish(FileNotFoundError(errno.ENOENT, "Data file closed before its sync", str(sync.file_path)))
        if sync_thread is not None:
            sync_thread.join()

    def __mark_unmarked_files(self) -> None:
        # main thread only. mark the canonical files which refused it, so that no version loads them after a crash or a
        # close. a file which is gone needs no mark.
        for file_path in list(self.__unmarked_files):
            if file_path.exists():
                try:
                    _mark_file_removed(file_path, file_locking=self.__file_locking)
                except OSError as e:
                    data_files_logger.debug("data file mark refused, to be tried again: %s, error %s", file_path, e)
                    continue
            self.__unmarked_files.remove(file_path)

    def __delete_files(self) -> bool:
        # main thread only. delete the files which this process no longer has open. a file whose delete is refused stays
        # for the next pass. return whether any file went.
        is_changed = False
        with self.__lock:
            deleted_files = list(self.__deleted_files.items())
        for open_path_key, file_path in deleted_files:
            if not HDF5FileEntry.is_path_open(file_path):
                try:
                    move_data_file(file_path, None, reason="deferred")
                except FileNotFoundError:
                    pass
                except OSError as e:
                    data_files_logger.debug("data file delete refused, to be tried again: %s, error %s", file_path, e)
                    continue
                with self.__lock:
                    self.__deleted_files.pop(open_path_key, None)
                is_changed = True
        return is_changed

    def __move_removed_files(self) -> bool:
        # main thread only. move the removed files which this process no longer has open into the trash. a file whose
        # move is refused stays in the removed files, to be tried again at the next pass. return whether any file went.
        is_changed = False
        with self.__lock:
            removed_files = list(self.__removed_files.values())
        for removed_file in removed_files:
            if not HDF5FileEntry.is_path_open(removed_file.file_path):
                # a pending file moves once synced, since its canonical file goes once it has moved.
                if removed_file.canonical_file_path is not None and removed_file.file_path.exists() and not self.__is_removed_file_synced(removed_file):
                    continue
                try:
                    if removed_file.file_path.exists():
                        move_data_file_to_trash(removed_file.file_path, removed_file.trash_file_path, reason=f"data item {removed_file.data_item_uuid} removed")
                except OSError as e:
                    data_files_logger.debug("removed data file move refused, to be tried again: %s, data item %s, error %s", removed_file.file_path, removed_file.data_item_uuid, e)
                    continue
                with self.__lock:
                    self.__removed_files.pop(removed_file.data_item_uuid, None)
                self.__removed_file_syncs.pop(removed_file.data_item_uuid, None)
                # the canonical file of a removed pending file goes once the pending file is in the trash.
                if removed_file.canonical_file_path is not None:
                    self.delete_file(removed_file.canonical_file_path)
                is_changed = True
        return is_changed

    # whether the removed file is synced. otherwise a sync is requested, unless one is waiting or running, and a refused
    # one is tried again. main thread only.
    def __is_removed_file_synced(self, removed_file: HDF5RemovedFile) -> bool:
        sync = self.__removed_file_syncs.get(removed_file.data_item_uuid)
        if sync is not None and sync.is_done and sync.error is None:
            return True
        if sync is None or sync.is_done:
            if sync is not None:
                data_files_logger.debug("removed pending file sync refused, to be tried again: %s, error %s", removed_file.file_path, sync.error)
            self.__removed_file_syncs[removed_file.data_item_uuid] = self.request_sync(removed_file.file_path)
        return False

    def __has_pending_files(self) -> bool:
        with self.__lock:
            return bool(self.__pending_files)

    def __make_refused_pending_files_ready(self) -> None:
        # main thread only, at close. a pending file left by its closed handler, whose replace is still refused, holds the
        # properties which describe its data and is synced, and is renamed to its ready file, which the next load
        # completes while the canonical file's stored modified time is the one it records.
        with self.__lock:
            pending_files = list(self.__pending_files.items())
        refused_pending_files = list[tuple[pathlib.Path, pathlib.Path, str]]()
        for pending_file_path, (canonical_file_path, handler_ref, canonical_modified) in pending_files:
            if handler_ref is None and pending_file_path.exists():
                # a pending file which the load made from a ready file is ready already.
                if pending_file_path == get_ready_file_path(canonical_file_path):
                    self.remove_pending_file(pending_file_path)
                # a time unknown when the file started pending isn't read here: a canonical file with no properties still
                # has none, and one which another program holds is as likely to refuse the read now. the next load
                # couldn't tell whether a ready file with no time is current, so the file stays a pending file.
                elif canonical_modified is None:
                    self.__warn_new_data_not_kept(pending_file_path, canonical_file_path)
                else:
                    refused_pending_files.append((pending_file_path, canonical_file_path, canonical_modified))
        # the rename comes after the tries of the replaces have used their time, so its tries have their own.
        recorded_file_paths = set[pathlib.Path]()
        deadline = time.monotonic() + _CLOSE_READY_RENAME_TIMEOUT
        while True:
            still_refused_pending_files = list[tuple[pathlib.Path, pathlib.Path, str]]()
            for refused_pending_file in refused_pending_files:
                pending_file_path, canonical_file_path, canonical_modified = refused_pending_file
                try:
                    if pending_file_path not in recorded_file_paths:
                        self.__record_canonical_modified(pending_file_path, canonical_modified)
                        recorded_file_paths.add(pending_file_path)
                    move_data_file(pending_file_path, get_ready_file_path(canonical_file_path), replace=True, reason="pending replacement refused at close")
                    self.remove_pending_file(pending_file_path)
                except OSError as e:
                    data_files_logger.debug("pending file not made ready at close: %s, error %s", pending_file_path, e)
                    still_refused_pending_files.append(refused_pending_file)
            refused_pending_files = still_refused_pending_files
            if not refused_pending_files or time.monotonic() >= deadline:
                break
            self.__close_retry_wait(_CLOSE_REPLACE_RETRY_INTERVAL)
        for pending_file_path, canonical_file_path, _ in refused_pending_files:
            self.__warn_new_data_not_kept(pending_file_path, canonical_file_path)

    def __warn_new_data_not_kept(self, pending_file_path: pathlib.Path, canonical_file_path: pathlib.Path) -> None:
        logging.getLogger("loader").warning(f"Unable to keep the new data of data file {canonical_file_path}. The next load discards {pending_file_path.name}, "
                                            "and the data file keeps its previous data.")

    def __record_canonical_modified(self, pending_file_path: pathlib.Path, canonical_modified: str) -> None:
        # main thread only, at close, as the last write to a pending file about to be renamed. the sync comes before the
        # rename, so that a power loss can't leave a ready file with a damaged record. raises OSError if the pending file
        # refuses.
        with h5py.File(pending_file_path, "r+", locking=None if self.__file_locking else False) as fp:
            fp["data"].attrs[_READY_CANONICAL_MODIFIED_ATTRIBUTE] = canonical_modified
        _sync_file(pending_file_path)

    def get_handlers_waiting_for_data_handle(self) -> list[HDF5Handler]:
        """Return the handlers whose pending file is ready to replace its canonical file but for its held data handle.

        Main thread only. A handler busy on another thread is left out until a later call.
        """
        with self.__lock:
            pending_files = list(self.__pending_files.values())
        return [handler for _, handler_ref, _ in pending_files if (handler := handler_ref() if handler_ref else None) and handler._is_waiting_for_data_handle_unless_busy()]

    def __complete_pending_files(self) -> bool:
        # main thread only. a handler completes its own replacement unless it is busy on another thread, and is then
        # skipped until the next pass. a replacement left by a closed handler completes once this process has closed both
        # files. return whether any replacement completed or was discarded.
        is_changed = False
        with self.__lock:
            pending_files = list(self.__pending_files.items())
        for pending_file_path, (canonical_file_path, handler_ref, _) in pending_files:
            if handler := handler_ref() if handler_ref else None:
                is_changed = handler.complete_pending_file_unless_busy() or is_changed
            elif not pending_file_path.exists():
                self.remove_pending_file(pending_file_path)
            elif handler_ref is not None:
                # the handler was dropped without closing, so the pending file may lack the properties which describe
                # its data, or may not be synced. it is discarded, and the canonical file keeps its previous data.
                data_files_logger.debug("pending replacement discarded, its handler dropped without closing: %s", pending_file_path)
                self.discard_pending_file(pending_file_path, canonical_file_path)
                is_changed = True
            else:
                try:
                    move_data_file(pending_file_path, canonical_file_path, replace=True, reason="pending replacement completed after its handler closed")
                except OSError as e:
                    data_files_logger.debug("pending replacement refused, to be tried again: %s, error %s", pending_file_path, e)
                    self.record_refused_replace(pending_file_path, canonical_file_path)
                    continue
                self.remove_pending_file(pending_file_path)
                is_changed = True
        return is_changed


class HDF5IdleFiles:
    """Current HDF5 files of handlers whose data handle nobody holds, closed once they stay idle for the delay.

    A handler queues its current file as idle when the last holder of its data handle lets go, and after it uses the
    file with no data handle held. close_idle_files closes a file which has been idle for the delay, and skips a file
    whose handler is busy, which stays idle for the next pass. The handler opens the file again when it is next used.

    The delay is in the units of the clock, which returns the current time.
    """

    def __init__(self, *, delay: float, clock: typing.Callable[[], float]) -> None:
        self.__delay = delay
        self.__clock = clock
        # the files queued as idle, with their handlers. files are queued from any thread; deque appends and pops are
        # thread safe. the queue is the only path between threads; the rest of the state is confined to the main thread.
        self.__idle_queue: collections.deque[tuple[HDF5Handler, HDF5FileEntry]] = collections.deque()
        # the idle files, oldest first, with their handlers and the time each was first seen idle. main thread only.
        self.__idle_files: dict[HDF5FileEntry, tuple[HDF5Handler, float]] = dict()

    @property
    def _idle_queue_length(self) -> int:
        # the number of queued files not yet taken in. for testing.
        return len(self.__idle_queue)

    @property
    def idle_count(self) -> int:
        """Return the number of files seen idle which are still open. Main thread only."""
        return sum(1 for file in self.__idle_files if file.is_open)

    def queue_idle(self, handler: HDF5Handler, file: HDF5FileEntry) -> None:
        # may be called on any thread, including from a finalizer. a file which is still queued is not queued again, so
        # that each file is queued at most once however often it is used.
        if not file.is_queued_idle:
            file.is_queued_idle = True
            self.__idle_queue.append((handler, file))

    def close_idle_files(self) -> bool:
        # main thread only. take in the queued files, then close the files idle for the delay, oldest first. a file
        # whose handler is in use on another thread is skipped and kept, as the oldest, for the next pass. return
        # whether any file closed.
        is_changed = False
        now = self.__clock()
        # take in the queued files without checking them; a file no longer idle is dropped when it comes up to close.
        while self.__idle_queue:
            handler, file = self.__idle_queue.popleft()
            file.is_queued_idle = False
            self.__idle_files.pop(file, None)
            self.__idle_files[file] = (handler, now)
        busy_files = dict[HDF5FileEntry, tuple[HDF5Handler, float]]()
        while oldest_entry := next(iter(self.__idle_files.items()), None):
            file, idle_entry = oldest_entry
            if now - idle_entry[1] < self.__delay:
                break
            self.__idle_files.pop(file)
            # a file which is no longer idle, or was already closed, is dropped without closing it.
            was_open = file.is_open
            if not idle_entry[0]._close_idle_file_unless_busy(file):
                busy_files[file] = idle_entry
            elif was_open and not file.is_open:
                is_changed = True
        # the busy files go back first, as the oldest.
        self.__idle_files = busy_files | self.__idle_files
        return is_changed

    def close(self) -> None:
        # main thread only. the handlers close their own files.
        self.__idle_queue.clear()
        self.__idle_files.clear()


class HDF5Handler(StorageHandler.StorageHandler):
    """The storage of a data item in an HDF5 file.

    The data item's file is its canonical file, at file_path, which reference returns. A change of shape or dtype writes
    a new pending file beside it (see find_pending_file_path), and the pending file then replaces the canonical file once
    it holds the properties which describe its data, a sync covers its writes, the canonical file isn't open in this
    process, and nobody holds the pending file's data handle. Until then the handler is pending: its data and properties
    are read from and written to the pending file, and the canonical file keeps the previous data and properties. A
    handler made with pending_file_path starts pending on that file, which may be a ready file, and canonical_modified is
    then the canonical file's stored modified time from get_stored_modified_json, or None if it couldn't be read. A handler
    made for a canonical file which its deferred files are waiting to delete keeps the file with its data and properties, and writes
    the properties it is given and its first data or reserve to a pending file whatever the shape.
    """
    count = 0  # useful for detecting leaks in tests

    def __init__(self, file_path: str | pathlib.Path, deferred_files: HDF5DeferredFiles | None = None, idle_files: HDF5IdleFiles | None = None, *,
                 pending_file_path: pathlib.Path | None = None, canonical_modified: str | None = None, file_locking: bool = True) -> None:
        self.__file_path = str(file_path)
        self.__lock = threading.RLock()
        # whether HDF5 locks the files this handler opens.
        self.__file_locking = file_locking
        # the canonical file, and the current file, which is the canonical file or a pending file which replaces it. the
        # current file and everything below are protected by the lock.
        self.__canonical_file_path = pathlib.Path(self.__file_path)
        self.__current_file_path = pending_file_path or self.__canonical_file_path
        # the number of the last pending file this handler made, so that each new one takes the next free number.
        self.__pending_file_index = 0
        # whether the pending file holds the properties which describe its data, which it needs before it replaces the
        # canonical file. a handler made with a pending file starts on one which holds them.
        self.__is_pending_file_described = pending_file_path is not None
        # the latest sync of the pending file, on which the replacement waits, with the write generations it covers, and
        # the latest generations found by a write through the handler or by a pass, with the time by the deferred files'
        # clock, so that a pending file still being written isn't synced.
        self.__pending_file_sync: tuple[HDF5FileSync, tuple[int, int]] | None = None
        self.__pending_file_last_write: tuple[tuple[int, int], float] | None = None
        # the generations of the writes into the current file through the handler, advanced under the lock, and through
        # its data handles, advanced by the handle which writes, so that the replacement can tell whether a write came
        # after a sync was requested.
        self.__handler_write_generation = 0
        self.__data_handle_write_generation = DataHandle.DataHandleWriteGeneration()
        # whether the canonical file is an earlier handler's file which was waiting to be deleted, as a storage type
        # change back to HDF5 finds it while its data is held. it is kept, and the first data written or reserved goes
        # into a pending file which replaces it, so that its holders keep reading the earlier data. the properties
        # written until then are kept here, as json, for the pending file, so that the kept file's properties go on
        # describing its data if a crash leaves it.
        self.__is_canonical_file_kept = pending_file_path is None and deferred_files is not None and deferred_files.keep_file(self.__canonical_file_path)
        self.__kept_file_properties: str | None = None
        self.__file = HDF5FileEntry(self.__current_file_path, locking=self.__file_locking)
        self.__dataset: typing.Any = None
        # the data handle handed out for the current dataset, while anyone holds it.
        self.__data_handle_ref: weakref.ReferenceType[DataHandle.DataHandle] | None = None
        # the deferred files of its storage system, which take the files this handler gives up while in use and the files
        # waiting to be renamed or deleted. a handler made without them makes its own, and closes them when it closes.
        self.__storage_deferred_files = deferred_files
        self.__deferred_files = deferred_files if deferred_files is not None else HDF5DeferredFiles()
        # the idle files of its storage system, which close the current file once nobody has used it for a while. a
        # handler made without them keeps its current file open until it closes, gives it up, or requests the sync of a
        # pending file whose data handle nobody holds.
        self.__idle_files = idle_files
        self._write_count = 0
        HDF5Handler.count += 1
        if pending_file_path is not None:
            self.__deferred_files.add_pending_file(pending_file_path, self.__canonical_file_path, self, canonical_modified)
            # the sync delay of a pending file which the handler starts on runs from the start.
            self.__record_pending_file_write()

    def close(self) -> None:
        HDF5Handler.count -= 1
        with self.__lock:
            # the data item writes no properties while its handler closes, so a described pending file holds all of them,
            # and its sync is waited for.
            is_synced = self.__is_pending_file_synced(wait=True)
            self.__release_file("handler closed")
            # a synced pending replacement is left to the deferred files, which complete it once this process has closed
            # both files.
            if self.__is_pending and is_synced:
                self.__deferred_files.leave_pending_file(self.__current_file_path)
            elif self.__is_pending:
                # a pending file without the properties which describe its data, or whose sync is refused, is discarded,
                # and the canonical file keeps the previous data and properties.
                self.__deferred_files.discard_pending_file(self.__current_file_path, self.__canonical_file_path)
        if self.__storage_deferred_files is None:
            self.__deferred_files.close()

    @property
    def factory(self) -> StorageHandler.StorageHandlerFactoryLike:
        return HDF5HandlerFactory(self.__storage_deferred_files, self.__idle_files, file_locking=self.__file_locking)

    @property
    def storage_handler_type(self) -> str:
        return "hdf5"

    # called before the file is moved; give up the current file.
    def prepare_move(self) -> None:
        with self.__lock:
            self.__release_file("given up")

    @property
    def reference(self) -> str:
        return self.__file_path

    # whether the current file is a pending file. under the lock.
    @property
    def __is_pending(self) -> bool:
        return self.__current_file_path != self.__canonical_file_path

    @property
    def is_valid(self) -> bool:
        return True

    def get_extension(self) -> str:
        return ".h5"

    # the properties as the json which a file stores.
    @staticmethod
    def __get_json_properties(properties: PersistentDictType) -> str:

        class JSONEncoder(json.JSONEncoder):
            def default(self, obj: typing.Any) -> typing.Any:
                if isinstance(obj, Geometry.IntPoint) or isinstance(obj, Geometry.IntSize) or isinstance(obj, Geometry.IntRect) or isinstance(obj, Geometry.FloatPoint) or isinstance(obj, Geometry.FloatSize) or isinstance(obj, Geometry.FloatRect):
                    return tuple(obj)
                else:
                    return json.JSONEncoder.default(self, obj)

        json_io = io.StringIO()
        json.dump(Utility.clean_dict(properties), json_io, cls=JSONEncoder)
        return json_io.getvalue()

    def __write_properties_to_dataset(self, properties: PersistentDictType) -> None:
        with self.__lock:
            assert self.__dataset is not None
            self.__dataset.attrs["properties"] = self.__get_json_properties(properties)

    def __ensure_dataset(self) -> None:
        with self.__lock:
            if self.__dataset is None:
                if "data" in self.__file.fp:
                    self.__dataset = self.__file.fp["data"]
                else:
                    self.__dataset = self.__file.fp.create_dataset("data", data=numpy.empty((0,)))
                    data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__current_file_path, self.__dataset.shape, self.__dataset.dtype)

    def write_data(self, data: _NDArray | DataHandle.DataHandle, data_descriptor: DataAndMetadata.DataDescriptor, file_datetime: datetime.datetime) -> None:
        with self.__lock:
            assert data is not None
            # handle three cases:
            #   1 - 'data' doesn't yet exist (require_dataset)
            #   2 - 'data' exists but is a different size, or is an earlier handler's (replace the file)
            #   3 - 'data' exists and is the same size (overwrite)
            if not "data" in self.__file.fp:
                # case 1
                chunks = get_write_chunk_shape_for_data(data.shape, data.dtype, data_descriptor)
                self.__dataset = self.__file.fp.require_dataset("data", shape=data.shape, dtype=data.dtype, chunks=chunks)
                data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__current_file_path, data.shape, data.dtype)
            else:
                if self.__dataset is None:
                    self.__dataset = self.__file.fp["data"]
                if self.__is_canonical_file_kept or self.__dataset.shape != data.shape or self.__dataset.dtype != data.dtype:
                    # case 2
                    self.__replace_file(data.shape, data.dtype, data_descriptor, data)
                    data_files_logger.debug("data written: %s, file type HDF5, shape %s, dtype %s", self.__current_file_path, data.shape, data.dtype)
                    self.__queue_if_idle()
                    return
            self.__handler_write_generation += 1
            # the data may be the current dataset's own handle, which a data item writes back after a partial update
            # through it.
            if data is self.__get_data_handle():
                # the handle already holds the data, written into the file through it, as a partial update is. this is on
                # the path of each partial update, so it adds no more than the logger's own level check.
                data_files_logger.debug("data written through its handle: %s, file type HDF5", self.__current_file_path)
            else:
                self.__dataset[:] = data
                self._write_count += 1
                data_files_logger.debug("data written: %s, file type HDF5, shape %s, dtype %s", self.__current_file_path, data.shape, data.dtype)
            self.__file.fp.flush()
            self.__record_pending_file_write()
            self.__queue_if_idle()

    def reserve_data(self, data_shape: DataAndMetadata.ShapeType, data_dtype: numpy.typing.DTypeLike, data_descriptor: DataAndMetadata.DataDescriptor, file_datetime: datetime.datetime) -> None:
        # reserve data of the given shape and dtype, filled with zeros
        with self.__lock:
            # existing data is replaced by a new file.
            if "data" in self.__file.fp:
                self.__replace_file(data_shape, data_dtype, data_descriptor, None)
            else:
                self.__handler_write_generation += 1
                chunks = get_write_chunk_shape_for_data(data_shape, data_dtype, data_descriptor)
                self.__dataset = self.__file.fp.require_dataset("data", shape=data_shape, dtype=data_dtype, fillvalue=0, chunks=chunks)
                self.__file.fp.flush()
                self.__record_pending_file_write()
                if data_files_logger.isEnabledFor(logging.DEBUG):
                    data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__current_file_path, data_shape, numpy.dtype(data_dtype))
            if data_files_logger.isEnabledFor(logging.DEBUG):
                data_files_logger.debug("data reserved: %s, file type HDF5, shape %s, dtype %s", self.__current_file_path, data_shape, numpy.dtype(data_dtype))
            self.__queue_if_idle()

    def __replace_file(self, data_shape: DataAndMetadata.ShapeType, data_dtype: numpy.typing.DTypeLike, data_descriptor: DataAndMetadata.DataDescriptor, data: _NDArray | DataHandle.DataHandle | None) -> None:
        # replace the current file with a new pending file holding data, or zeros when data is None, and the current
        # properties. the current file is given up, and a superseded pending file is discarded.
        # the replacement completes in a later pass of the deferred files, once the properties which describe the new
        # data are written into the pending file. under the lock, on the writing thread.
        if self.__dataset is None:
            self.__dataset = self.__file.fp["data"]
        json_properties = self.__dataset.attrs.get("properties", "")
        # nothing writes the canonical file while a file is pending, so a superseded pending file's time still holds.
        canonical_modified: str | None
        if self.__is_pending:
            canonical_modified = self.__deferred_files.get_pending_canonical_modified(self.__current_file_path)
        else:
            canonical_properties = _parse_properties(json_properties)
            canonical_modified = get_stored_modified_json(canonical_properties) if canonical_properties is not None else None
        # a kept file's properties describe its own data, and the pending file takes the ones written since.
        if self.__kept_file_properties is not None:
            json_properties = self.__kept_file_properties
        pending_file_path, self.__pending_file_index = find_pending_file_path(self.__canonical_file_path, self.__pending_file_index + 1)
        pending_file = HDF5FileEntry(pending_file_path, locking=self.__file_locking)
        try:
            chunks = get_write_chunk_shape_for_data(data_shape, data_dtype, data_descriptor)
            dataset = pending_file.fp.require_dataset("data", shape=data_shape, dtype=data_dtype, fillvalue=0, chunks=chunks)
            if data is not None:
                dataset[:] = data
                self._write_count += 1
            dataset.attrs["properties"] = json_properties
            pending_file.fp.flush()
        except Exception:
            # the current file stays as it was; the incomplete pending file goes.
            pending_file.close("its write failed")
            self.__deferred_files.delete_file(pending_file_path)
            raise
        superseded_file_path = self.__current_file_path if self.__is_pending else None
        self.__release_file("replaced")
        if superseded_file_path is not None:
            data_files_logger.debug("pending replacement superseded: %s", superseded_file_path)
            self.__deferred_files.discard_pending_file(superseded_file_path, self.__canonical_file_path)
        self.__current_file_path = pending_file_path
        self.__file = pending_file
        self.__dataset = dataset
        self.__is_pending_file_described = False
        self.__pending_file_sync = None
        self.__pending_file_last_write = None
        self.__is_canonical_file_kept = False
        self.__kept_file_properties = None
        self.__deferred_files.add_pending_file(pending_file_path, self.__canonical_file_path, self, canonical_modified)
        data_files_logger.debug("pending replacement started: %s, replaces %s, shape %s, dtype %s", pending_file_path, self.__canonical_file_path, data_shape, dataset.dtype)

    # complete a pending replacement when a sync covers the writes into the described pending file, the canonical file is
    # not open in this process and the pending file's data handle is not held: the pending file is closed and replaces
    # the canonical file, which becomes the current file again, opened when next used. a refused replacement leaves the
    # handler pending, to be tried again by a later pass of the deferred files. return whether it completed. under the
    # lock.
    def __complete_pending_file(self) -> bool:
        if not self.__is_pending_file_synced() or self.__get_data_handle() is not None or HDF5FileEntry.is_path_open(self.__canonical_file_path):
            return False
        self.__dataset = None
        self.__data_handle_ref = None
        self.__file.close("replacing its canonical file")
        try:
            move_data_file(self.__current_file_path, self.__canonical_file_path, replace=True, reason="pending replacement completed")
        except OSError as e:
            data_files_logger.debug("pending replacement refused, to be tried again: %s, error %s", self.__current_file_path, e)
            self.__deferred_files.record_refused_replace(self.__current_file_path, self.__canonical_file_path)
            return False
        self.__deferred_files.remove_pending_file(self.__current_file_path)
        self.__current_file_path = self.__canonical_file_path
        self.__file = HDF5FileEntry(self.__current_file_path, locking=self.__file_locking)
        self.__pending_file_sync = None
        self.__pending_file_last_write = None
        return True

    # request the sync of a described pending file which is due, and complete the pending replacement if it can, unless
    # another thread holds the lock; return whether it completed.
    def complete_pending_file_unless_busy(self) -> bool:
        if self.__lock.acquire(blocking=False):
            try:
                return self.__complete_pending_file()
            finally:
                self.__lock.release()
        return False

    # whether the pending replacement waits only for the holders of its data handle to let go, unless another thread
    # holds the lock, which counts as not waiting. main thread only.
    def _is_waiting_for_data_handle_unless_busy(self) -> bool:
        if self.__lock.acquire(blocking=False):
            try:
                return self.__is_pending_file_synced() and self.__get_data_handle() is not None and not HDF5FileEntry.is_path_open(self.__canonical_file_path)
            finally:
                self.__lock.release()
        return False

    # whether the current file is a pending file which holds the properties which describe its data, and a finished sync
    # on the sync thread covers every write into it which finished before the sync was requested. a call once the file
    # has had no writes for the sync delay requests the sync, and a later call finds it finished; with wait, it requests
    # the sync at once and waits for it. a refused sync is requested again. under the lock.
    def __is_pending_file_synced(self, *, wait: bool = False) -> bool:
        if not self.__is_pending or not self.__is_pending_file_described:
            return False
        is_requested = False
        while True:
            # the generations are taken before h5py flushes the file, so that a write after them asks for another sync.
            write_generations = self.__get_write_generations()
            # a sync which doesn't cover the writes so far is followed by another.
            if self.__pending_file_sync is not None and self.__pending_file_sync[1] == write_generations:
                sync = self.__pending_file_sync[0]
                if wait:
                    sync.wait()
                if not sync.is_done:
                    return False
                if sync.error is None:
                    return True
                data_files_logger.debug("pending file sync refused: %s, error %s", self.__current_file_path, sync.error)
                if is_requested:
                    return False
            # a file still being written is synced once it has had no writes for the sync delay, so that a file written
            # continually isn't synced again and again. a write through a data handle is timed from the call which finds
            # it.
            if not wait:
                if self.__pending_file_last_write is None or self.__pending_file_last_write[0] != write_generations:
                    self.__pending_file_last_write = (write_generations, self.__deferred_files.get_time())
                if not self.__deferred_files.is_sync_due(self.__pending_file_last_write[1]):
                    return False
            # a file whose data handle nobody holds is closed first, so that the sync covers what h5py writes as it
            # closes the file, and the sync thread's open is the first since the writes. on Windows that first open is
            # slow, and the next use of the file would otherwise wait for it on the main thread.
            if self.__get_data_handle() is None:
                self.__dataset = None
                self.__data_handle_ref = None
                self.__file.close("syncing")
            else:
                self.__file.fp.flush()
            self.__pending_file_sync = (self.__deferred_files.request_sync(self.__current_file_path), write_generations)
            is_requested = True
            if not wait:
                return False

    # the generations of the writes through the handler and through its data handles. under the lock.
    def __get_write_generations(self) -> tuple[int, int]:
        return self.__handler_write_generation, self.__data_handle_write_generation.value

    # record a write through the handler into a pending file, with its time, from which the sync delay runs. under the
    # lock.
    def __record_pending_file_write(self) -> None:
        if self.__is_pending:
            self.__pending_file_last_write = (self.__get_write_generations(), self.__deferred_files.get_time())

    def write_properties(self, properties: PersistentDictType, file_datetime: datetime.datetime) -> None:
        with self.__lock:
            # a kept file keeps its own properties, and these wait for the pending file.
            if self.__is_canonical_file_kept:
                self.__kept_file_properties = self.__get_json_properties(properties)
                return
            self.__handler_write_generation += 1
            self.__ensure_dataset()
            self.__write_properties_to_dataset(properties)
            self.__file.fp.flush()
            self.__record_pending_file_write()
            # properties written into a pending file describe its data. the replacement completes in a later pass, once
            # all of them are written.
            if self.__is_pending:
                self.__is_pending_file_described = True
            self.__queue_if_idle()

    def read_properties(self) -> PersistentDictType:
        with self.__lock:
            # properties waiting for the pending file of a kept file are the handler's current ones.
            if self.__kept_file_properties is not None:
                return typing.cast(PersistentDictType, json.loads(self.__kept_file_properties))
            self.__ensure_dataset()
            json_properties = self.__dataset.attrs.get("properties", "")
            self.__queue_if_idle()
            return typing.cast(PersistentDictType, json.loads(json_properties))

    def read_data(self) -> DataHandle.DataHandle | None:
        with self.__lock:
            self.__ensure_dataset()
            if self.__dataset.shape == (0, ):
                self.__queue_if_idle()
                return None
            data_handle = self.__get_data_handle()
            if data_handle is None:
                data_handle = DataHandle.DataHandle(self.__dataset, self.__data_handle_write_generation)
                self.__data_handle_ref = weakref.ref(data_handle)
                # when the last holder lets go, the file becomes idle. the finalizer queues this file entry, so it does
                # nothing once the file is given up.
                if self.__idle_files is not None:
                    weakref.finalize(data_handle, self.__idle_files.queue_idle, self, self.__file)
            return data_handle

    # the data handle handed out for the current dataset, if it is still held.
    def __get_data_handle(self) -> DataHandle.DataHandle | None:
        return self.__data_handle_ref() if self.__data_handle_ref else None

    # queue the current file as idle when no data handle is held, so that it closes once nobody has used it for a while.
    # called under the lock after using the file.
    def __queue_if_idle(self) -> None:
        if self.__idle_files is not None and self.__get_data_handle() is None:
            self.__idle_files.queue_idle(self, self.__file)

    # whether the file is still the current file, open, and its data handle not held. any thread.
    def _is_file_idle(self, file: HDF5FileEntry) -> bool:
        with self.__lock:
            return file is self.__file and file.is_open and self.__get_data_handle() is None

    # close the file if it is still idle, unless another thread holds the lock; return whether the lock was free. a data
    # handle is only made under the lock, so a reader on another thread either holds a handle before this, and the file
    # stays open, or opens the file again after it. main thread only; it never waits for this lock.
    def _close_idle_file_unless_busy(self, file: HDF5FileEntry) -> bool:
        if not self.__lock.acquire(blocking=False):
            return False
        try:
            if self._is_file_idle(file):
                self.__dataset = None
                self.__data_handle_ref = None
                file.close("idle")
            return True
        finally:
            self.__lock.release()

    def remove(self) -> None:
        # a file this process still has open, because its data is held, is deleted once it is closed.
        with self.__lock:
            self.__release_file("given up")
            if self.__is_pending:
                self.__deferred_files.remove_pending_file(self.__current_file_path)
                self.__deferred_files.delete_file(self.__current_file_path)
                self.__current_file_path = self.__canonical_file_path
                self.__file = HDF5FileEntry(self.__current_file_path, locking=self.__file_locking)
            if self.__canonical_file_path.is_file():
                self.__deferred_files.delete_file(self.__canonical_file_path)

    def release_for_removal(self, trash_file_path: pathlib.Path) -> bool:
        """Give up the file for a removal which can be undone; return whether it is left to the deferred files.

        A file whose data handle is held is marked removed and retired, so that it stays open at its path for its holders,
        who can no longer write it, and moves to trash_file_path after the last holder lets go. Otherwise the file is
        closed, for the caller to move.

        A pending handler is always left to the deferred files. Its pending file moves to trash_file_path once it is closed
        and synced. Its canonical file, which holds the previous data, is marked removed, and deleted once the pending file
        has moved and the canonical file is closed.
        """
        with self.__lock:
            if self.__is_pending:
                self.__release_pending_for_removal(trash_file_path)
                return True
            if self.__get_data_handle() is None:
                self.__release_file("given up")
                return False
            self.__deferred_files.add_removed_file(self.__mark_removed(trash_file_path))
            self.__release_file("given up")
            return True

    def retire_removed_file(self, trash_file_path: pathlib.Path) -> None:
        """Mark the file removed and leave it to the deferred files, which move it to trash_file_path at a later pass.

        For a removal whose move into the trash is refused. The file is closed again afterwards.
        """
        with self.__lock:
            removed_file = self.__mark_removed(trash_file_path)
            self.__release_file("given up")
            self.__deferred_files.add_removed_file(removed_file)

    # mark the current file removed in its properties, leaving the rest of them as they are, and return its removed file.
    def __mark_removed(self, trash_file_path: pathlib.Path) -> HDF5RemovedFile:
        self.__ensure_dataset()
        data_item_uuid = _mark_dataset_removed(self.__dataset)
        self.__file.fp.flush()
        data_files_logger.debug("data file marked removed: %s, data item %s, file type HDF5", self.__current_file_path, data_item_uuid)
        return HDF5RemovedFile(data_item_uuid, pathlib.Path(self.__file_path), trash_file_path)

    # give up a pending handler's files for a removal which can be undone; afterwards the handler is no longer pending.
    # under the lock, on the main thread.
    def __release_pending_for_removal(self, trash_file_path: pathlib.Path) -> None:
        # the canonical file is marked removed, so that no version loads it after a crash. it is deleted once the pending
        # file has moved into the trash, so that a restore before then finds both files. another program may refuse the
        # mark, as it does the replace which this removal follows, and the deferred files then try it again.
        if self.__canonical_file_path.is_file():
            try:
                _mark_file_removed(self.__canonical_file_path, file_locking=self.__file_locking)
            except OSError as e:
                data_files_logger.debug("data file mark refused, to be tried again: %s, error %s", self.__canonical_file_path, e)
                self.__deferred_files.add_unmarked_file(self.__canonical_file_path)
        # the pending file is hidden, so it is not marked removed. it moves into the trash under the canonical name, where
        # a restore finds it, once it is closed and synced.
        self.__ensure_dataset()
        properties = typing.cast(PersistentDictType, json.loads(self.__dataset.attrs.get("properties", "")))
        pending_file_path = self.__current_file_path
        canonical_modified = self.__deferred_files.get_pending_canonical_modified(pending_file_path)
        self.__release_file("given up")
        self.__deferred_files.remove_pending_file(pending_file_path)
        self.__current_file_path = self.__canonical_file_path
        self.__file = HDF5FileEntry(self.__current_file_path, locking=self.__file_locking)
        self.__pending_file_sync = None
        self.__pending_file_last_write = None
        self.__deferred_files.add_removed_file(HDF5RemovedFile(uuid.UUID(properties["uuid"]), pending_file_path, trash_file_path, self.__canonical_file_path, canonical_modified))

    # give up the current file. a file whose data handle is held is retired, so that its holders keep reading its data
    # but can no longer write it. otherwise the file is closed, for the reason, which goes into the data files log.
    # either way the handler continues with a new file entry at the current path, opened when next used, so that the
    # earlier file, if queued as idle, no longer matches.
    def __release_file(self, reason: str) -> None:
        data_handle = self.__get_data_handle()
        self.__dataset = None
        self.__data_handle_ref = None
        if data_handle is not None:
            data_handle.mark_replaced()
            self.__deferred_files.retire(self.__file, data_handle)
        else:
            self.__file.close(reason)
        self.__file = HDF5FileEntry(self.__current_file_path, locking=self.__file_locking)


class HDF5HandlerFactory(StorageHandler.StorageHandlerFactoryLike):

    def __init__(self, deferred_files: HDF5DeferredFiles | None = None, idle_files: HDF5IdleFiles | None = None, *, file_locking: bool = True) -> None:
        # the deferred and idle files of the storage system which makes handlers with this factory, if any, and whether HDF5
        # locks the files its handlers open.
        self.__deferred_files = deferred_files
        self.__idle_files = idle_files
        self.__file_locking = file_locking

    def get_storage_handler_type(self) -> str:
        return "hdf5"

    def is_matching(self, file_path: str) -> bool:
        if file_path.endswith(".h5") and os.path.exists(file_path):
            return True
        return False

    def make(self, file_path: pathlib.Path) -> StorageHandler.StorageHandler:
        return HDF5Handler(self.make_path(file_path), self.__deferred_files, self.__idle_files, file_locking=self.__file_locking)

    def make_pending(self, file_path: pathlib.Path, pending_file_path: pathlib.Path, canonical_modified: str | None) -> HDF5Handler:
        """Make a handler for the canonical file at file_path which starts pending on the file at pending_file_path.

        canonical_modified is the canonical file's stored modified time from get_stored_modified_json, or None if it
        couldn't be read.
        """
        return HDF5Handler(self.make_path(file_path), self.__deferred_files, self.__idle_files, pending_file_path=pending_file_path, canonical_modified=canonical_modified,
                           file_locking=self.__file_locking)

    def check_file_writable(self, file_path: pathlib.Path) -> None:
        """Raise OSError if the HDF5 file at file_path refuses being opened for writing."""
        with h5py.File(file_path, "r+", locking=None if self.__file_locking else False):
            pass

    def read_file_properties(self, file_path: pathlib.Path) -> PersistentDictType | None:
        """Return the properties in the HDF5 file at file_path, opening it read only, or None if they can't be read."""
        return _read_file_properties(file_path, file_locking=self.__file_locking)

    def is_file_refused(self, file_path: pathlib.Path) -> bool:
        """Return whether another program refuses this process the HDF5 file at file_path, as opposed to the file being damaged.

        Windows refuses a rename of the file to its own name while another program has it open, and h5py raises a plain
        OSError there for a file which another program has locked, as it does for a damaged one. On macOS and Linux the
        rename succeeds, and h5py raises PermissionError for a refused access or BlockingIOError for an HDF5 file lock
        which another program holds. A missing file isn't refused. The file must not be open in this process.
        """
        try:
            os.rename(file_path, file_path)
            with h5py.File(file_path, "r", locking=None if self.__file_locking else False):
                pass
        except (PermissionError, BlockingIOError):
            return True
        except OSError:
            return False
        return False

    def read_ready_canonical_modified(self, ready_file_path: pathlib.Path) -> str | None:
        """Return the canonical file's stored modified time as the ready file recorded it, or None if the ready file has no record or its properties can't be read."""
        return _read_ready_canonical_modified(ready_file_path, file_locking=self.__file_locking)

    def make_path(self, file_path: pathlib.Path) -> str:
        return str(file_path.with_suffix(self.get_extension()))

    def get_extension(self) -> str:
        return ".h5"


def _create_dataset(data_group: typing.Any, dataset_id: str, data: numpy.typing.NDArray[typing.Any]) -> typing.Any:
    data_shape = data.shape

    # create the dataset, preallocate space.
    ds = data_group.create_dataset(dataset_id, shape=data_shape, dtype=data.dtype)

    # search for the chunk size by iterating backwards through the shape and finding the
    # largest chunk size that is less than 64MB.
    index_count = 0
    chunk_size = 1
    for n in reversed(data_shape):
        if chunk_size * n > 64 * 1024 * 1024:
            break
        index_count += 1
        chunk_size *= n

    # iterate over the remaining dimensions so that we can write the data in chunks.
    for index in numpy.ndindex(*data_shape[:len(data_shape) - index_count]):
        selection = tuple(index) + (slice(None),) * index_count
        ds[selection] = data[selection]

    return ds


class HDFImportExportDriver:
    def __init__(self) -> None:
        pass

    def read_data(self, file_path: pathlib.Path, storage_handler_provider: StorageHandler.StorageHandlerProvider) -> StorageHandler.StorageHandlerImportData:
        storage_handlers = list[StorageHandler.StorageHandler]()
        uuid_map = dict[uuid.UUID, uuid.UUID]()
        items = list[PersistentDictType]()
        fp = h5py.File(file_path, "r")
        if "data" in fp:
            data_group = fp["data"]
            for key in sorted(data_group.keys()):
                ds = data_group[key]
                data_item_uuid = uuid.uuid4()
                data_item_properties = json.loads(ds.attrs["properties"])
                data_item_data = ds
                if "uuid" in data_item_properties:
                    uuid_map[uuid.UUID(data_item_properties["uuid"])] = data_item_uuid
                data_item_properties["uuid"] = str(data_item_uuid)
                is_sequence = data_item_properties.get("is_sequence", False)
                collection_dimension_count = data_item_properties.get("collection_dimension_count", 0)
                datum_dimension_count = data_item_properties.get("datum_dimension_count", 0)
                data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
                assert data_descriptor.expected_dimension_count == len(ds.shape)
                storage_handler_attributes = StorageHandler.StorageHandlerAttributes(
                    data_item_uuid,
                    DateTime.utcnow(),
                    data_item_properties.get("session_id", None),
                    ds.nbytes
                )
                storage_handler = storage_handler_provider.make_storage_handler(storage_handler_attributes)
                storage_handler.write_data(data_item_data, data_descriptor, storage_handler_attributes.created_local)
                storage_handler.write_properties(data_item_properties, storage_handler_attributes.created_local)
                storage_handlers.append(storage_handler)
        if "index" in fp:
            index_group = fp["index"]
            for key in sorted(index_group.attrs.keys()):
                items.append(json.loads(index_group.attrs[key]))
        fp.close()
        return StorageHandler.StorageHandlerImportData(storage_handlers, uuid_map, items)

    def write_display_item(self, path: pathlib.Path, items: typing.Sequence[StorageHandler.StorageHandlerExportItem]) -> None:
        path.unlink(missing_ok=True)
        fp = h5py.File(path, "a")
        index_group = fp.create_group("index")
        data_group = fp.create_group("data")
        data_index = 0
        for index, item in enumerate(items):
            index_group.attrs["1"] = json.dumps(item.write_to_dict())
            for data_item in item.data_items:
                ds = _create_dataset(data_group, str(data_index), data_item.data)
                ds.attrs["properties"] = json.dumps(data_item.write_to_dict())
                data_index += 1
        fp.close()


class HDFImportExportDriverFactory:
    driver_id = "nhdf-io-handler"
    title = "NData HDF"
    extensions = ["nhdf"]

    def make_import_export_driver(self) -> HDFImportExportDriver:
        return HDFImportExportDriver()


Registry.register_component(HDFImportExportDriverFactory(), {"import-export-driver-factory"})


"""
Architectural Decision Records.

ADR 2024-02-19. Do not use Single Writer Multiple Reader (SWMR) mode for HDF5 files. This requires the writing process
to set the flag and we don't have control of the writers outside of our process. In addition, the docs curently say that
SWMR is not supported on Windows.
"""
