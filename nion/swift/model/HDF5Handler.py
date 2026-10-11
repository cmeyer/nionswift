"""
    A module for handle .h5 files for Swift.
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import logging
import os
import pathlib
import sys
import threading
import typing
import uuid
import weakref

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

# the log of data file management: data files created, written, moved and deleted. its records are at DEBUG, so it is off
# by default. turn it on with logging.getLogger("nion.swift.data_files").setLevel(logging.DEBUG), for example from the
# console or a debugger. its records then go to the app's stderr, not to the Output panel.
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

# Windows refuses to delete or rename a file held open, so there a held file is closed under its holders.
_retire_held_files = sys.platform != "win32"


class HDF5FileEntry:
    """An HDF5 file at a path, opened when first used and owned by one handler.

    Each handler opens its own file, so closing it affects only that handler's data. The file may be opened again after
    it is closed. The count of open files is kept for detecting unclosed files in tests.
    """
    _open_count = 0
    __open_count_lock = threading.Lock()

    def __init__(self, path: pathlib.Path) -> None:
        self.__lock = threading.RLock()
        self.__path = path
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

    def open(self) -> None:
        with self.__lock:
            if not self.__fp:
                self.__path.parent.mkdir(parents=True, exist_ok=True)
                self.__fp = h5py.File(self.__path, "a")
                with HDF5FileEntry.__open_count_lock:
                    HDF5FileEntry._open_count += 1

    def close(self) -> None:
        with self.__lock:
            if self.__fp:
                self.__fp.close()
                with HDF5FileEntry.__open_count_lock:
                    HDF5FileEntry._open_count -= 1
            self.__fp = None

    @property
    def is_open(self) -> bool:
        with self.__lock:
            return self.__fp is not None


class HDF5RetiredFiles:
    """Open HDF5 files which their handlers have replaced while a data handle over the file is held.

    A retired file stays open so that the holders of its data handle keep reading its data. When the last holder drops
    the handle, a finalizer queues the file, and close_released_files closes it on the main thread. Close closes every
    retired file, whether or not its holders have gone.
    """

    def __init__(self) -> None:
        # the retired files and the finalizers of their data handles. files are retired on any thread; the lock protects
        # the dict.
        self.__lock = threading.Lock()
        self.__retired_files: dict[HDF5FileEntry, weakref.finalize[[HDF5FileEntry], DataHandle.DataHandle]] = dict()
        # the retired files whose last holder has gone. a finalizer may run on any thread, inside unrelated code, so it
        # only appends; deque appends and pops are thread safe, and only the main thread pops.
        self.__released_files: collections.deque[HDF5FileEntry] = collections.deque()

    def retire(self, file: HDF5FileEntry, data_handle: DataHandle.DataHandle) -> None:
        # may be called on any thread.
        with self.__lock:
            self.__retired_files[file] = weakref.finalize(data_handle, self.__released_files.append, file)

    def close_released_files(self) -> None:
        # main thread only.
        while self.__released_files:
            file = self.__released_files.popleft()
            with self.__lock:
                self.__retired_files.pop(file, None)
            file.close()

    def close(self) -> None:
        # main thread only.
        with self.__lock:
            retired_files = self.__retired_files
            self.__retired_files = dict()
        for file, finalizer in retired_files.items():
            finalizer.detach()
            file.close()
        self.__released_files.clear()


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

    def queue_idle(self, handler: HDF5Handler, file: HDF5FileEntry) -> None:
        # may be called on any thread, including from a finalizer. a file which is still queued is not queued again, so
        # that each file is queued at most once however often it is used.
        if not file.is_queued_idle:
            file.is_queued_idle = True
            self.__idle_queue.append((handler, file))

    def close_idle_files(self) -> None:
        # main thread only. take in the queued files, then close the files idle for the delay, oldest first. a file
        # whose handler is in use on another thread is skipped and kept, as the oldest, for the next pass.
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
            if not idle_entry[0]._close_idle_file_unless_busy(file):
                busy_files[file] = idle_entry
        # the busy files go back first, as the oldest.
        self.__idle_files = busy_files | self.__idle_files

    def close(self) -> None:
        # main thread only. the handlers close their own files.
        self.__idle_queue.clear()
        self.__idle_files.clear()


class HDF5Handler(StorageHandler.StorageHandler):
    count = 0  # useful for detecting leaks in tests

    def __init__(self, file_path: str | pathlib.Path, retired_files: HDF5RetiredFiles | None = None, idle_files: HDF5IdleFiles | None = None) -> None:
        self.__file_path = str(file_path)
        self.__lock = threading.RLock()
        self.__file = HDF5FileEntry(pathlib.Path(self.__file_path))
        self.__dataset: typing.Any = None
        # the data handle handed out for the current dataset, while anyone holds it.
        self.__data_handle_ref: weakref.ReferenceType[DataHandle.DataHandle] | None = None
        # files this handler replaces while their data handle is held go to the retired files of its storage system,
        # which closes them after their last holder. a handler made without them closes its retired files when it closes.
        self.__storage_retired_files = retired_files
        self.__retired_files = retired_files if retired_files is not None else HDF5RetiredFiles()
        # the idle files of its storage system, which close the current file once nobody has used it for a while. a
        # handler made without them keeps its current file open until it closes or gives it up.
        self.__idle_files = idle_files
        self._write_count = 0
        HDF5Handler.count += 1

    def close(self) -> None:
        HDF5Handler.count -= 1
        with self.__lock:
            self.__release_file()
        if self.__storage_retired_files is None:
            self.__retired_files.close()

    @property
    def factory(self) -> StorageHandler.StorageHandlerFactoryLike:
        return HDF5HandlerFactory(self.__storage_retired_files, self.__idle_files)

    @property
    def storage_handler_type(self) -> str:
        return "hdf5"

    # called before the file is moved; give up the current file.
    def prepare_move(self) -> None:
        with self.__lock:
            self.__release_file()

    @property
    def reference(self) -> str:
        return self.__file_path

    @property
    def is_valid(self) -> bool:
        return True

    def get_extension(self) -> str:
        return ".h5"

    def __write_properties_to_dataset(self, properties: PersistentDictType) -> None:
        with self.__lock:
            assert self.__dataset is not None

            class JSONEncoder(json.JSONEncoder):
                def default(self, obj: typing.Any) -> typing.Any:
                    if isinstance(obj, Geometry.IntPoint) or isinstance(obj, Geometry.IntSize) or isinstance(obj, Geometry.IntRect) or isinstance(obj, Geometry.FloatPoint) or isinstance(obj, Geometry.FloatSize) or isinstance(obj, Geometry.FloatRect):
                        return tuple(obj)
                    else:
                        return json.JSONEncoder.default(self, obj)

            json_io = io.StringIO()
            json.dump(Utility.clean_dict(properties), json_io, cls=JSONEncoder)
            json_str = json_io.getvalue()

            self.__dataset.attrs["properties"] = json_str

    def __ensure_dataset(self) -> None:
        with self.__lock:
            if self.__dataset is None:
                if "data" in self.__file.fp:
                    self.__dataset = self.__file.fp["data"]
                else:
                    self.__dataset = self.__file.fp.create_dataset("data", data=numpy.empty((0,)))
                    data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__file_path, self.__dataset.shape, self.__dataset.dtype)

    def write_data(self, data: _NDArray | DataHandle.DataHandle, data_descriptor: DataAndMetadata.DataDescriptor, file_datetime: datetime.datetime) -> None:
        with self.__lock:
            assert data is not None
            json_properties = None
            # handle three cases:
            #   1 - 'data' doesn't yet exist (require_dataset)
            #   2 - 'data' exists but is a different size (delete, then require_dataset)
            #   3 - 'data' exists and is the same size (overwrite)
            if not "data" in self.__file.fp:
                # case 1
                chunks = get_write_chunk_shape_for_data(data.shape, data.dtype, data_descriptor)
                self.__dataset = self.__file.fp.require_dataset("data", shape=data.shape, dtype=data.dtype, chunks=chunks)
                data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__file_path, data.shape, data.dtype)
            else:
                if self.__dataset is None:
                    self.__dataset = self.__file.fp["data"]
                if self.__dataset.shape != data.shape or self.__dataset.dtype != data.dtype:
                    # case 2
                    json_properties = self.__dataset.attrs.get("properties", "")
                    self.__release_file()
                    os.remove(self.__file_path)
                    data_files_logger.debug("data file deleted: %s, file type HDF5, reason data of another shape or dtype written", self.__file_path)
                    chunks = get_write_chunk_shape_for_data(data.shape, data.dtype, data_descriptor)
                    self.__dataset = self.__file.fp.require_dataset("data", shape=data.shape, dtype=data.dtype, chunks=chunks)
                    data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__file_path, data.shape, data.dtype)
            # the data may be the current dataset's own handle, which a data item writes back after a partial update
            # through it.
            if data is self.__get_data_handle():
                # the handle already holds the data, written into the file through it, as a partial update is. this is on
                # the path of each partial update, so it adds no more than the logger's own level check.
                data_files_logger.debug("data written through its handle: %s, file type HDF5", self.__file_path)
            else:
                self.__dataset[:] = data
                self._write_count += 1
                data_files_logger.debug("data written: %s, file type HDF5, shape %s, dtype %s", self.__file_path, data.shape, data.dtype)
            if json_properties is not None:
                self.__dataset.attrs["properties"] = json_properties
            self.__file.fp.flush()
            self.__queue_if_idle()

    def reserve_data(self, data_shape: DataAndMetadata.ShapeType, data_dtype: numpy.typing.DTypeLike, data_descriptor: DataAndMetadata.DataDescriptor, file_datetime: datetime.datetime) -> None:
        # reserve data of the given shape and dtype, filled with zeros
        with self.__lock:
            json_properties = None
            # first read existing properties and then close existing data set and file.
            if "data" in self.__file.fp:
                if self.__dataset is None:
                    self.__dataset = self.__file.fp["data"]
                json_properties = self.__dataset.attrs.get("properties", "")
                self.__release_file()
                os.remove(self.__file_path)
                data_files_logger.debug("data file deleted: %s, file type HDF5, reason data reserved", self.__file_path)
                self.__file.open()
            # reserve the data
            chunks = get_write_chunk_shape_for_data(data_shape, data_dtype, data_descriptor)
            self.__dataset = self.__file.fp.require_dataset("data", shape=data_shape, dtype=data_dtype, fillvalue=0, chunks=chunks)
            if json_properties is not None:
                self.__dataset.attrs["properties"] = json_properties
            self.__file.fp.flush()
            if data_files_logger.isEnabledFor(logging.DEBUG):
                dtype = numpy.dtype(data_dtype)
                data_files_logger.debug("data file created: %s, file type HDF5, shape %s, dtype %s", self.__file_path, data_shape, dtype)
                data_files_logger.debug("data reserved: %s, file type HDF5, shape %s, dtype %s", self.__file_path, data_shape, dtype)
            self.__queue_if_idle()

    def write_properties(self, properties: PersistentDictType, file_datetime: datetime.datetime) -> None:
        with self.__lock:
            self.__ensure_dataset()
            self.__write_properties_to_dataset(properties)
            self.__file.fp.flush()
            self.__queue_if_idle()

    def read_properties(self) -> PersistentDictType:
        with self.__lock:
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
                data_handle = DataHandle.DataHandle(self.__dataset)
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
                file.close()
            return True
        finally:
            self.__lock.release()

    def remove(self) -> None:
        with self.__lock:
            self.__release_file()
            if os.path.isfile(self.__file_path):
                os.remove(self.__file_path)
                data_files_logger.debug("data file deleted: %s, file type HDF5", self.__file_path)

    # give up the current file. a file whose data handle is held is retired, so that its holders keep reading its data
    # but can no longer write it. otherwise the file is closed. either way the handler continues with a new file entry
    # at its path, opened when next used, so that the earlier file, if queued as idle, no longer matches.
    def __release_file(self) -> None:
        data_handle = self.__get_data_handle()
        self.__dataset = None
        self.__data_handle_ref = None
        if data_handle is not None and _retire_held_files:
            data_handle.mark_replaced()
            self.__retired_files.retire(self.__file, data_handle)
        else:
            self.__file.close()
        self.__file = HDF5FileEntry(pathlib.Path(self.__file_path))


class HDF5HandlerFactory(StorageHandler.StorageHandlerFactoryLike):

    def __init__(self, retired_files: HDF5RetiredFiles | None = None, idle_files: HDF5IdleFiles | None = None) -> None:
        # the retired and idle files of the storage system which makes handlers with this factory, if any.
        self.__retired_files = retired_files
        self.__idle_files = idle_files

    def get_storage_handler_type(self) -> str:
        return "hdf5"

    def is_matching(self, file_path: str) -> bool:
        if file_path.endswith(".h5") and os.path.exists(file_path):
            return True
        return False

    def make(self, file_path: pathlib.Path) -> StorageHandler.StorageHandler:
        return HDF5Handler(self.make_path(file_path), self.__retired_files, self.__idle_files)

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
