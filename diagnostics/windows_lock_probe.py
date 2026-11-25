"""Report what h5py raises when another program holds an HDF5 file, so that the data-handle branch counts the right errors as a refusal.

Usage: python windows_lock_probe.py
It writes a small file in the system temporary folder, then for each kind of holder starts a second Python process which holds the
file, and opens the file for writing from this process. It prints the exception type, errno and winerror for each, and removes the file.
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import h5py
import numpy

HOLDERS = {
    "h5py read (HDF5 locking on)": "import h5py, sys, time; f = h5py.File(sys.argv[1], 'r'); print('held', flush=True); time.sleep(30)",
    "h5py write (HDF5 locking on)": "import h5py, sys, time; f = h5py.File(sys.argv[1], 'r+'); print('held', flush=True); time.sleep(30)",
    "Python open, read": "import sys, time; f = open(sys.argv[1], 'rb'); print('held', flush=True); time.sleep(30)",
}


def describe(e: BaseException) -> str:
    return f"{type(e).__name__}, errno {getattr(e, 'errno', None)}, winerror {getattr(e, 'winerror', None)}: {e}"


def main() -> None:
    print(f"python {sys.version.split()[0]}, h5py {h5py.__version__}, HDF5 {h5py.version.hdf5_version}, {sys.platform}")
    with tempfile.TemporaryDirectory() as directory:
        file_path = pathlib.Path(directory) / "data_probe.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=numpy.zeros((4, 4)))
        # with no holder, a rename to its own name must succeed, or the undo check would refuse every undo.
        try:
            os.rename(file_path, file_path)
            print("no holder: rename to its own name succeeded")
        except Exception as e:
            print(f"no holder: rename to its own name raised {describe(e)}")
        for name, code in HOLDERS.items():
            holder = subprocess.Popen([sys.executable, "-c", code, str(file_path)], stdout=subprocess.PIPE, text=True)
            try:
                assert holder.stdout is not None
                holder.stdout.readline()
                for mode in ("r+", "r"):
                    try:
                        with h5py.File(file_path, mode):
                            pass
                        print(f"{name}: open {mode!r} succeeded")
                    except Exception as e:
                        print(f"{name}: open {mode!r} raised {describe(e)}")
                try:
                    os.replace(file_path, file_path.with_suffix(".moved"))
                    os.replace(file_path.with_suffix(".moved"), file_path)
                    print(f"{name}: rename succeeded")
                except Exception as e:
                    print(f"{name}: rename raised {describe(e)}")
                # the branch's undo check renames a file at its path to its own name, which must refuse while the file is held.
                try:
                    os.rename(file_path, file_path)
                    print(f"{name}: rename to its own name succeeded")
                except Exception as e:
                    print(f"{name}: rename to its own name raised {describe(e)}")
            finally:
                holder.kill()
                holder.wait()
                time.sleep(0.5)


if __name__ == "__main__":
    main()
