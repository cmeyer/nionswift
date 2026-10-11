import contextlib
import datetime
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import threading
import unittest
import uuid

# third party libraries
import numpy
import numpy.typing

# local libraries
from nion.data import DataAndMetadata
from nion.swift.model import HDF5Handler
from nion.swift.model import Cache


class TestHDF5Handler(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_write_chunk_shape_for_data(self):

        data_shapes_and_dtypes = [((512, 512, 130, 130), 'float32', DataAndMetadata.DataDescriptor(False, 2, 2)),
                                  ((971, 971, 130, 130), 'float32', DataAndMetadata.DataDescriptor(False, 2, 2)),
                                  ((1024, 1024, 130, 260), 'uint8', DataAndMetadata.DataDescriptor(True, 2, 1)),
                                  ((1024, 1024, 1030), 'int16', DataAndMetadata.DataDescriptor(False, 2, 1)),
                                  ((1024, 1024, 1032), 'float32', DataAndMetadata.DataDescriptor(False, 2, 1)),
                                  ((512, 512), 'float32', DataAndMetadata.DataDescriptor(False, 0, 2)),
                                  ((int(1e12),), 'int64', DataAndMetadata.DataDescriptor(False, 0, 1)),
                                  ((10, 1024, 1024, 1030), 'float32', DataAndMetadata.DataDescriptor(True, 2, 1)),
                                  ((3, 123, 17, 31, 29), 'float32', DataAndMetadata.DataDescriptor(False, 2, 2))]
        expected_chunk_shapes = [(1, 58, 32, 32),
                                 (1, 58, 32, 32),
                                 (1, 1, 130, 260),
                                 (1, 466, 257),
                                 (1, 232, 258),
                                 None,
                                 (30000,),
                                 (1, 1, 233, 257),
                                 (1, 1, 17, 31, 29)]

        for shape_dtype_descriptor, expected_chunk_shape in zip(data_shapes_and_dtypes, expected_chunk_shapes):
            with self.subTest(shape=shape_dtype_descriptor[0], dtype=shape_dtype_descriptor[1], data_descriptor=shape_dtype_descriptor[2]):
                chunk_shape = HDF5Handler.get_write_chunk_shape_for_data(*shape_dtype_descriptor)

                if expected_chunk_shape is None:
                    self.assertIsNone(chunk_shape)
                else:
                    self.assertSequenceEqual(chunk_shape, expected_chunk_shape)

    def test_hdf5_handler_basic_functionality(self):
        now = datetime.datetime.now()
        current_working_directory = pathlib.Path.cwd()
        data_dir = current_working_directory / "__Test"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        Cache.db_make_directory_if_needed(data_dir)
        try:
            h = HDF5Handler.HDF5Handler(os.path.join(data_dir, "abc.h5"))
            with contextlib.closing(h):
                p = {u"abc": 1, u"def": u"bcd", u"uuid": str(uuid.uuid4())}
                # write properties
                h.write_properties(p, now)
                self.assertEqual(h.read_properties(), p)
                self.assertIsNone(h.read_data())
                # write data
                h.write_data(numpy.zeros((4,4), dtype=numpy.float64), DataAndMetadata.DataDescriptor(False, 0, 2), now)
                self.assertEqual(h.read_properties(), p)
                d = h.read_data()
                self.assertEqual(d.shape, (4, 4))
                self.assertEqual(d.dtype, numpy.float64)
                # rewrite data
                h.write_data(numpy.zeros((12,12), dtype=numpy.float32), DataAndMetadata.DataDescriptor(False, 0, 2), now)
                self.assertEqual(h.read_properties(), p)
                d = h.read_data()
                self.assertEqual(d.shape, (12, 12))
                self.assertEqual(d.dtype, numpy.float32)
                # rewrite properties
                h.write_properties(p, now)
                self.assertEqual(h.read_properties(), p)
                d = h.read_data()
                self.assertEqual(d.shape, (12, 12))
                self.assertEqual(d.dtype, numpy.float32)
                # reserve data
                h.reserve_data((3, 15), numpy.float32, DataAndMetadata.DataDescriptor(False, 0, 2), now)
                self.assertEqual(h.read_properties(), p)
                d = h.read_data()
                self.assertEqual(d.shape, (3, 15))
                self.assertEqual(d.dtype, numpy.float32)
                self.assertTrue(numpy.allclose(d, 0))
        finally:
            #logging.debug("rmtree %s", data_dir)
            shutil.rmtree(data_dir)

    def test_data_handle_gives_the_values_of_the_stored_array(self):
        stored = numpy.random.default_rng(41).random((6, 9, 13)).astype(numpy.float32)
        mask = numpy.zeros(stored.shape, dtype=bool)
        mask[1:3, 2:7, 4:12] = True
        with tempfile.TemporaryDirectory() as directory:
            handler = HDF5Handler.HDF5Handler(pathlib.Path(directory) / "data.h5")
            with contextlib.closing(handler):
                handler.write_data(stored, DataAndMetadata.DataDescriptor(False, 1, 2), datetime.datetime.now())
                data_handle = handler.read_data()
                assert data_handle is not None
                self.assertEqual(stored.shape, data_handle.shape)
                self.assertEqual(stored.dtype, data_handle.dtype)
                self.assertEqual(stored.ndim, data_handle.ndim)
                self.assertEqual(stored.size, data_handle.size)
                self.assertEqual(stored.nbytes, data_handle.nbytes)
                self.assertEqual(len(stored), len(data_handle))
                self.assertTrue(numpy.array_equal(stored[2:5, 1:8], data_handle[2:5, 1:8]))
                self.assertTrue(numpy.array_equal(stored[..., 3], data_handle[..., 3]))
                self.assertEqual(stored[1, 2, 3], data_handle[1, 2, 3])
                self.assertTrue(numpy.array_equal(stored[[1, 4]], data_handle[[1, 4]]))
                self.assertTrue(numpy.array_equal(stored[mask], data_handle[mask]))
                self.assertEqual(len(stored), len(list(data_handle)))
                for stored_row, handle_row in zip(stored, data_handle):
                    self.assertTrue(numpy.array_equal(stored_row, handle_row))
                self.assertTrue(numpy.array_equal(stored, numpy.asarray(data_handle)))
                self.assertTrue(numpy.allclose(numpy.sum(stored, axis=1), numpy.sum(data_handle, axis=1)))
                as_float64 = data_handle.astype(numpy.float64)[2:4]
                self.assertEqual(numpy.dtype(numpy.float64), as_float64.dtype)
                self.assertTrue(numpy.array_equal(stored[2:4].astype(numpy.float64), as_float64))
                self.assertTrue(numpy.array_equal(stored, data_handle.copy()))

    def test_slice_of_data_handle_is_independent_of_the_stored_data(self):
        stored = numpy.random.default_rng(41).random((6, 9)).astype(numpy.float32)
        with tempfile.TemporaryDirectory() as directory:
            handler = HDF5Handler.HDF5Handler(pathlib.Path(directory) / "data.h5")
            with contextlib.closing(handler):
                handler.write_data(stored, DataAndMetadata.DataDescriptor(False, 0, 2), datetime.datetime.now())
                data_handle = handler.read_data()
                assert data_handle is not None
                data_slice = data_handle[1:4, 2:7]
                self.assertIsInstance(data_slice, numpy.ndarray)
                data_slice[:] = 0
                self.assertTrue(numpy.array_equal(stored[1:4, 2:7], data_handle[1:4, 2:7]))

    def test_writing_another_files_data_handle_copies_the_data(self):
        stored = numpy.random.default_rng(41).random((6, 9)).astype(numpy.float32)
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
        with tempfile.TemporaryDirectory() as directory:
            handler = HDF5Handler.HDF5Handler(pathlib.Path(directory) / "data.h5")
            other_handler = HDF5Handler.HDF5Handler(pathlib.Path(directory) / "other.h5")
            with contextlib.closing(handler), contextlib.closing(other_handler):
                handler.write_data(stored, data_descriptor, datetime.datetime.now())
                other_handler.write_data(numpy.zeros_like(stored), data_descriptor, datetime.datetime.now())
                other_write_count = other_handler._write_count
                data_handle = handler.read_data()
                assert data_handle is not None
                other_handler.write_data(data_handle, data_descriptor, datetime.datetime.now())
                self.assertEqual(other_write_count + 1, other_handler._write_count)
                self.assertTrue(numpy.array_equal(stored, numpy.asarray(other_handler.read_data())))

    def test_closing_idle_files_skips_a_file_in_use_on_another_thread_and_closes_it_once_free(self):
        # data which another thread writes, and which waits for an event when the write reads it, so that the write holds
        # the handler busy for as long as the test needs, as a long write does.
        class WaitingData:
            def __init__(self, data: numpy.ndarray, reading_event: threading.Event, release_event: threading.Event) -> None:
                self.data = data
                self.shape = data.shape
                self.dtype = data.dtype
                self.reading_event = reading_event
                self.release_event = release_event

            def __array__(self, dtype: numpy.typing.DTypeLike = None, copy: bool | None = None) -> numpy.ndarray:
                self.reading_event.set()
                # the timeout only keeps a failing test from hanging.
                self.release_event.wait(10.0)
                return self.data.astype(dtype) if dtype is not None else self.data

        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
        clock_time = [100.0]
        idle_files = HDF5Handler.HDF5IdleFiles(delay=5.0, clock=lambda: clock_time[0])
        with tempfile.TemporaryDirectory() as directory:
            handler = HDF5Handler.HDF5Handler(pathlib.Path(directory) / "data.h5", idle_files=idle_files)
            with contextlib.closing(handler):
                open_count = HDF5Handler.HDF5FileEntry._open_count
                handler.write_data(numpy.zeros((5, 7), numpy.float32), data_descriptor, datetime.datetime.now())
                # the pass takes the file in as idle, so that it has been idle for the delay at the pass below.
                idle_files.close_idle_files()
                reading_event = threading.Event()
                release_event = threading.Event()
                waiting_data = WaitingData(numpy.ones((5, 7), numpy.float32), reading_event, release_event)
                thread = threading.Thread(target=handler.write_data, args=(waiting_data, data_descriptor, datetime.datetime.now()))
                thread.start()
                try:
                    self.assertTrue(reading_event.wait(10.0))
                    # the file has been idle for the delay, but its handler is in use, so the file stays open and the
                    # main thread goes on without waiting for the write.
                    clock_time[0] += 5.0
                    idle_files.close_idle_files()
                    self.assertTrue(thread.is_alive())
                    self.assertEqual(open_count + 1, HDF5Handler.HDF5FileEntry._open_count)
                finally:
                    release_event.set()
                    thread.join()
                # the write used the file, so once the write is done the file closes when it has been idle for the delay
                # since then.
                idle_files.close_idle_files()
                self.assertEqual(open_count + 1, HDF5Handler.HDF5FileEntry._open_count)
                clock_time[0] += 5.0
                idle_files.close_idle_files()
                self.assertEqual(open_count, HDF5Handler.HDF5FileEntry._open_count)
                self.assertTrue(numpy.array_equal(numpy.ones((5, 7), numpy.float32), numpy.asarray(handler.read_data())))

    def test_using_an_unheld_file_many_times_queues_it_idle_once(self):
        properties = {"title": "title", "uuid": str(uuid.uuid4())}
        idle_files = HDF5Handler.HDF5IdleFiles(delay=5.0, clock=lambda: 100.0)
        with tempfile.TemporaryDirectory() as directory:
            handler = HDF5Handler.HDF5Handler(pathlib.Path(directory) / "data.h5", idle_files=idle_files)
            with contextlib.closing(handler):
                handler.write_data(numpy.zeros((5, 7), numpy.float32), DataAndMetadata.DataDescriptor(False, 0, 2), datetime.datetime.now())
                handler.write_properties(properties, datetime.datetime.now())
                for _ in range(100):
                    self.assertEqual(properties, handler.read_properties())
                # each use queues the file as idle, since no data handle is held; it stays queued until taken in.
                self.assertEqual(1, idle_files._idle_queue_length)
                # once the file is taken in, the next use queues it again.
                idle_files.close_idle_files()
                self.assertEqual(0, idle_files._idle_queue_length)
                handler.read_properties()
                self.assertEqual(1, idle_files._idle_queue_length)

    @unittest.skipUnless(sys.platform == "win32" or hasattr(os, "chflags"), "Linux has no lock which refuses the move of a file.")
    def test_move_of_data_file_refused_by_a_lock_on_it_leaves_no_copy_at_the_new_path(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = pathlib.Path(directory) / "data.h5"
            file_path.write_bytes(bytes(8))
            new_file_path = pathlib.Path(directory) / "moved.h5"
            # Windows refuses the move while another program has the file open, and macOS while the file is immutable.
            with open(file_path, "rb"):
                if hasattr(os, "chflags"):
                    os.chflags(file_path, stat.UF_IMMUTABLE)
                try:
                    with self.assertRaises(OSError):
                        HDF5Handler.move_data_file(file_path, new_file_path)
                finally:
                    if hasattr(os, "chflags"):
                        os.chflags(file_path, 0)
            self.assertFalse(new_file_path.exists())

    def test_pending_replacement_is_skipped_while_its_handler_is_in_use_on_another_thread_and_completes_once_free(self):
        # data which another thread writes, and which waits for an event when the write reads it, so that the write holds
        # the handler busy for as long as the test needs, as a long write does.
        class WaitingData:
            def __init__(self, data: numpy.ndarray, reading_event: threading.Event, release_event: threading.Event) -> None:
                self.data = data
                self.shape = data.shape
                self.dtype = data.dtype
                self.reading_event = reading_event
                self.release_event = release_event

            def __array__(self, dtype: numpy.typing.DTypeLike = None, copy: bool | None = None) -> numpy.ndarray:
                self.reading_event.set()
                # the timeout only keeps a failing test from hanging.
                self.release_event.wait(10.0)
                return self.data.astype(dtype) if dtype is not None else self.data

        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
        deferred_files = HDF5Handler.HDF5DeferredFiles()
        with tempfile.TemporaryDirectory() as directory:
            file_path = pathlib.Path(directory) / "data.h5"
            handler = HDF5Handler.HDF5Handler(file_path, deferred_files)
            with contextlib.closing(handler):
                handler.write_data(numpy.zeros((5, 7), numpy.float32), data_descriptor, datetime.datetime.now())
                data_handle = handler.read_data()
                # the reshape goes to a pending file while the old data is held. the old file closes once the old data is
                # let go, but the new data is held, so the replacement waits. the properties which describe the new data
                # follow it, as a data item writes them.
                handler.write_data(numpy.zeros((6, 4), numpy.float32), data_descriptor, datetime.datetime.now())
                handler.write_properties({"data_shape": [6, 4], "data_dtype": "float32"}, datetime.datetime.now())
                new_data_handle = handler.read_data()
                data_handle = None
                deferred_files.close_released_files()
                self.assertEqual(2, len(list(file_path.parent.iterdir())))
                reading_event = threading.Event()
                release_event = threading.Event()
                waiting_data = WaitingData(numpy.ones((6, 4), numpy.float32), reading_event, release_event)
                thread = threading.Thread(target=handler.write_data, args=(waiting_data, data_descriptor, datetime.datetime.now()))
                thread.start()
                try:
                    self.assertTrue(reading_event.wait(10.0))
                    # the new data is let go, but the handler is in use, so the replacement waits and the main thread goes
                    # on without waiting for the write.
                    new_data_handle = None
                    deferred_files.close_released_files()
                    self.assertTrue(thread.is_alive())
                    self.assertEqual(2, len(list(file_path.parent.iterdir())))
                finally:
                    release_event.set()
                    thread.join()
                deferred_files.close_released_files()
                self.assertEqual(["data.h5"], sorted(path.name for path in file_path.parent.iterdir()))
                self.assertTrue(numpy.array_equal(numpy.ones((6, 4), numpy.float32), numpy.asarray(handler.read_data())))
            deferred_files.close()
