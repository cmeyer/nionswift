# Other programs holding data files, on Windows

Read `diagnostics\PROMPT.md` first, for the setup, the rules and what the branch does, then do this task instead of its
steps. The rules there apply here too:
- don't change, commit or push repository files, and don't install anything;
- write your scripts and results only in `test-results\`;
- run anything long in the background.

This task doesn't time anything, so it needs no pinning and no quiet machine.

## Why Windows

On Windows, another program which has a data file open refuses a rename of it, and often a write, a delete or a move.
Such programs include:
- a cloud sync client;
- an antivirus scanner;
- an h5py script;
- a second copy of Nion Swift.

The branch renames files to replace them, and moves them into and out of the trash. So it has code which handles a
refusal. In each case it tries again later, refuses the operation cleanly, or reports it. On macOS a holder doesn't
block a rename, so this code is only really exercised on Windows.

The test suite covers it with locks taken in the test process itself, for example `_FileLock` in
`nion\swift\test\Storage_test.py`. This task checks it end to end with a real second process.

## What the branch promises when a file is held

- **A reshape or a reserve whose replace is refused** leaves the old data in place and retries at each pass. When the
  holder lets go, the replace completes.
- **A replace refused for 60 s** is reported once to the window, as a stuck replacement. `ProjectStorageSettings` has a
  delay setting for this; find it in `nion\swift\model\FileStorageSystem.py`.
- **Closing the project while a replace is refused** retries for a short time. Then it renames the pending file to a
  hidden ready file, and the next load completes the replace. If the item's file was changed by an earlier version in
  between, the ready file is discarded.
- **An undo of a removal whose files are held** is refused. The command stays on the undo stack, a dialog says so, and a
  later undo after the holder lets go restores the item with its data. A redo is the same.
- **A removal whose file can't be marked** retries the mark at each pass, and warns at close if it still can't.
- **Nothing raises out of these operations,** and saved data is never lost.

## What to do

1. **A holder script** in `test-results\`, run as a second process, which holds a given file until told to stop. Write
   it in three kinds:
   - h5py open for reading, with HDF5 file locking on;
   - h5py open for writing;
   - a plain Python `open(path, "rb")`.

   `diagnostics\windows_lock_probe.py` shows how each kind refuses operations on Windows.

2. **Scenario scripts,** one per promise above. Each runs a headless document controller on a file-stored project,
   starts the holder on the right file at the right moment, drives the operation, releases the holder, and checks the
   outcome. To set up the project and drive the controller, read the refused-undo and refused-replace tests in
   `nion\swift\test\Storage_test.py`, and `performance\hdf5_replace.py`. The outcome is:
   - the items and their data;
   - the undo stack;
   - the files left in the data folder;
   - whether any exception escaped.

   Run each scenario with each holder kind, and run it on master too for comparison. Master has none of this handling,
   so expect it to raise or lose data in some cases.

3. **Look further:** anything the branch does which a holder could refuse and these scenarios miss, such as idle files
   closing and reopening, or moving a file out of the trash.

## Report

Write `test-results\REPORT-holders.md` in plain language. For each scenario and holder kind, give:
- what the branch did, and what master did;
- whether the promise held;
- for every failure, the exception or the lost data, the files left behind, and a short script which reproduces it.

Keep the scripts in `test-results\`. When done, tell the user to zip `test-results` as `test-results-laptop-holders.zip`
and bring it back.
