"""
    A module for the handle a data item hands out for data stored in a file.
"""
from __future__ import annotations

import threading
import typing

import numpy
import numpy.typing


_NDArray = numpy.typing.NDArray[typing.Any]


class DataHandleWriteGeneration:
    """The generation of the writes through the data handles which share it, which each write advances.

    Only one handle of a storage handler accepts writes at a time, and it advances the generation under its write lock.
    Another thread reads the generation without a lock, since reading an int is atomic.
    """

    def __init__(self) -> None:
        self.value = 0


class DataHandle:
    """A handle to a data item's data stored in a file, standing in for the array.

    A data item stored in HDF5 hands out a data handle as its data, in place of the file's dataset. The handle has the
    interface an h5py dataset has (shape, dtype, ndim, size, nbytes, len, indexing, assignment by index, iteration,
    conversion to numpy and a lazy astype), and copy, which returns the whole array as a numpy array.

    Indexing returns numpy arrays which are copies, so a slice of a handle owns its data and keeps no link to the file.
    Assignment by index writes into the file. Converting the handle to numpy, or copying it, reads the whole array.

    When the data item's data is replaced while the handle is held, the handle keeps reading the earlier data but
    refuses writes, raising ValueError, as numpy does for a read-only array.

    Numpy operators and array methods other than these (arithmetic, reshape, ravel, T, real, flags) are not available,
    as on an h5py dataset; convert to numpy first.

    The handle is the only route to the stored dataset, which it does not expose.
    """

    # the dataset is an h5py dataset, which has no type information. the write generation, if given, advances with each
    # write.
    def __init__(self, dataset: typing.Any, write_generation: DataHandleWriteGeneration | None = None) -> None:
        self.__dataset = dataset
        # whether the data item's data has been replaced. the write lock orders each write against the mark.
        self.__replaced = False
        self.__write_lock = threading.Lock()
        self.__write_generation = write_generation

    def __repr__(self) -> str:
        return f"<DataHandle {self.shape} {self.dtype}>"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.__dataset.shape)

    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        return typing.cast(numpy.dtype[typing.Any], self.__dataset.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(numpy.prod(self.shape, dtype=numpy.int64))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("Attempt to take len() of a scalar data handle.")
        return shape[0]

    # the key is anything h5py accepts as an index; the result is a numpy array or, for a scalar index, a numpy scalar.
    def __getitem__(self, key: typing.Any) -> typing.Any:
        return self.__dataset[key]

    # the key and value are anything h5py accepts for assignment by index.
    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        with self.__write_lock:
            if self.__replaced:
                raise ValueError("The data item's data was replaced, so the write to its earlier data did not happen.")
            try:
                self.__dataset[key] = value
            finally:
                # a write which raises may have reached the file in part, so it advances the generation too.
                if self.__write_generation is not None:
                    self.__write_generation.value += 1

    # iterate over the first axis; each item is what indexing returns, a numpy array or a numpy scalar.
    def __iter__(self) -> typing.Iterator[typing.Any]:
        for index in range(len(self)):
            yield self[index]

    # copy is forwarded only when it is given, so that an h5py whose __array__ takes no copy argument still converts.
    def __array__(self, dtype: numpy.typing.DTypeLike | None = None, /, copy: bool | None = None) -> _NDArray:
        if copy is None:
            return typing.cast(_NDArray, self.__dataset.__array__(dtype))
        return typing.cast(_NDArray, self.__dataset.__array__(dtype, copy=copy))

    def astype(self, dtype: numpy.typing.DTypeLike) -> DataHandle | DataHandleAsTypeView:
        """Return a view which reads the data converted to dtype, or this handle if dtype is its own.

        As with an h5py dataset, the view reads only what is indexed.
        """
        new_dtype = numpy.dtype(dtype)
        if new_dtype == self.dtype:
            return self
        return DataHandleAsTypeView(self, new_dtype)

    def copy(self) -> _NDArray:
        """Return the whole array as a numpy array."""
        return self.__array__()

    def mark_replaced(self) -> None:
        """Mark the handle as over data which its storage handler has replaced, so that it refuses writes.

        It waits for a write in progress, a single assignment by index, so the wait is bounded.
        """
        with self.__write_lock:
            self.__replaced = True

    # read the selection converted to dtype, for DataHandleAsTypeView. the key is anything h5py accepts as an index; the
    # result is a numpy array or, for a scalar index, a numpy scalar.
    def _get_item_as_type(self, key: typing.Any, dtype: numpy.dtype[typing.Any]) -> typing.Any:
        return self.__dataset.__getitem__(key, new_dtype=dtype)


class DataHandleAsTypeView:
    """A view of a data handle which reads its data converted to another dtype, as an h5py dataset's astype does."""

    def __init__(self, data_handle: DataHandle, dtype: numpy.dtype[typing.Any]) -> None:
        self.__data_handle = data_handle
        self.__dtype = dtype

    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        return self.__dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__data_handle.shape

    def __len__(self) -> int:
        return len(self.__data_handle)

    # the key is anything h5py accepts as an index; the result is a numpy array or, for a scalar index, a numpy scalar.
    def __getitem__(self, key: typing.Any) -> typing.Any:
        return self.__data_handle._get_item_as_type(key, self.__dtype)

    def __array__(self, dtype: numpy.typing.DTypeLike | None = None, /, copy: bool | None = None) -> _NDArray:
        return self.__data_handle.__array__(dtype if dtype is not None else self.__dtype, copy=copy)
