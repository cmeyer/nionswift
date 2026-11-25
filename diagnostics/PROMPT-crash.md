# Crash tests of the data-handle branch on Windows

Read `diagnostics\PROMPT.md` first, for the setup, the rules and what the branch does, then do this task instead of its
steps. The rules there apply here too:
- don't change, commit or push repository files, and don't install anything;
- write your scripts and results only in `test-results\`;
- run anything long in the background.

This task doesn't time anything, so it needs no pinning and no quiet machine.

## The guarantee to test

A crash may lose new data, which was being written when the process died. It must never lose or damage data which was
already saved. After any crash, the next load of the project must give, for every data item which existed before the
crash:
- data and properties which belong together, either the item's old pair or its new pair, and never one file's data with
  another's properties;
- data which reads back whole.

A data item which was being removed may be either present or gone. No other data item may be lost.

The branch exists to give this guarantee where master doesn't. Master deletes an item's file before it writes data of a
new shape, so a crash in between loses saved data. The branch writes the new data to a hidden pending file, syncs it, and
renames it over the item's file. On Windows, the rename, the sync (`os.fsync`, that is `FlushFileBuffers`) and file
locking all behave differently from macOS, where the crash checks so far ran. Those checks copied the project's files at
chosen points; this task kills a running process instead.

## What to do

1. **A child script** in `test-results\`, run as its own process. It opens a file-stored project in a folder you give it,
   using `nion.swift.test.TestContext.FileProfileContext` or `nion.swift.test.Storage_test.create_temp_profile_context`
   with `large_format_size=0` so that every item is stored in HDF5, and `no_remove=True` so that the folder stays. Read
   `performance\hdf5_replace.py` for how a script sets these up.

   It loops over operations on a few data items of a few MiB each:
   - set data of a new shape (a reshape);
   - set data of the same shape;
   - reserve data, then write frames into it as an acquisition does;
   - partial updates;
   - remove an item and add a new one;
   - periodically, run the document controller's periodic, so that pending files are synced and replaced.

   The idle delay is 5 s, and a replace completes only after it. Wait at times, so that kills also land during the
   sync, the rename and the later reopening.

   Make every item's data identify its own generation, for example by writing the generation number into its first
   value, and record the same number in the item's metadata or title. Then a load can check that data and properties
   belong together.

2. **A parent script.** It starts the child, waits a random time from 0 to 10 s, kills it the hard way (`Stop-Process
   -Force`, or `TerminateProcess` through Python's `subprocess` `kill()`), then opens the project in a new process and
   checks the guarantee for every item. It records what each item held and which operation the child was in, if the
   child logs that to a file before each step.

   Repeat it many times, at least 200 kills, on a fresh project each time or carrying one project forward, as you judge
   best. Then do the same against master, at the commit `windows_run.ps1` measures as master, for comparison: master is
   expected to lose data, and the branch isn't.

3. **Clean up checks.** After each load, also check whether hidden files (names starting with `.`) are left in the data
   folder, and whether they disappear at the next load.

## Report

Write `test-results\REPORT-crash.md` in plain language:
- how the child and the parent work;
- how many kills, during which operations;
- for the branch and for master, how many loads kept the guarantee and how many broke it;
- for every break: what was lost or mismatched, the operation, and the files left in the data folder. Keep a copy of
  that project's folder in `test-results\` as evidence.

When done, tell the user to zip `test-results` as `test-results-laptop-crash.zip` and bring it back.
