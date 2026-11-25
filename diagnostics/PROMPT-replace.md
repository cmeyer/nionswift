# Why the first write after a replace is slower on the branch

Read `diagnostics\PROMPT.md` first, for the setup, the rules and what the branch does, then do this task instead of its
steps. The rules there apply here too: don't change, commit or push repository files, don't install anything, pin timings
to processors 0 and 1, alternate master and the branch over three rounds, and run anything long in the background.

## What is known

A previous run on this laptop found the data-handle branch the same as master for `frame_write`, `load_read` and
`rewrite`. It was slower only where a file is replaced:
- `reshape`: about +30 to 40 ms;
- `rewrite_after_reshape`: about +40 to 55 ms. This is the first same-shape `set_data` on a 256 MiB HDF5 item after a
  reshape has settled.

The task is to find out why `rewrite_after_reshape` is slower, and whether that can be avoided.

On the branch, a reshape:
1. writes a hidden pending file;
2. syncs it with `_sync_file` in `nion\swift\model\HDF5Handler.py`, which on Windows calls `os.fsync`, that is
   `FlushFileBuffers`;
3. renames it over the item's file with `os.replace`;
4. then reopens the item's file when the handler next uses it.

On master, the reshape writes into the item's file in place.

## Steps

1. **A plain Python probe,** with h5py and no nionswift, written in `test-results\`. It times one full write of a
   256 MiB float32 dataset into each of these files:
   - a file left open;
   - a file closed and reopened;
   - a file just created;
   - a file just created, fsync'd and renamed over another;
   - the same, but without the fsync.

   Repeat each case 10 times, alternating, pinned to processors 0 and 1, with the work folder `C:\perftemp`. The results
   show whether the cost is the reopen, the fsync, the rename, or a new file itself.
2. **The branch with one step changed.** Using the `HDF5Replace` class in `performance\hdf5_replace.py`, run
   `rewrite_after_reshape` on the branch with one step changed at a time, by monkeypatching in your own script:
   - no sync: `_sync_file` does nothing;
   - the file opened once more right after the replace completes, before the rewrite is timed.

   Compare each with the unpatched branch and with master, pinned and alternating over three rounds.
3. **A way around it.** If one step is the cause, look for a way to avoid it which doesn't weaken what the branch
   guarantees:
   - old data stays safe after a crash, including a power loss: the sync before the rename exists for this;
   - old data stays readable while something holds it.

   If you find one, describe the change and measure it as a monkeypatch. Don't edit the repository. A change which skips
   or delays the sync weakens the first guarantee; say so if that's the only way.

## Report

Write `test-results\REPORT-replace.md` in plain language:
- each experiment, how it was run, its numbers, and what it shows;
- your conclusion, with how sure you are.

When done, tell the user to zip `test-results` as `test-results-laptop-replace.zip` and bring it back.
