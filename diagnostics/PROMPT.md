# Windows diagnostics for the data-handle branch

You are on a Windows laptop with a nionswift clone checked out on the diagnostics branch. Your job is to measure whether
the data-handle branch is slower than master on this machine, and if it is, to find which part of the branch causes it.
Report what you find; don't change the branch's code.

## What the branch is

The diagnostics branch is the data-handle branch plus one commit adding this `diagnostics` folder.

- **The data-handle branch** changes how nionswift stores HDF5 data. A data item hands out a `DataHandle` in place of the
  h5py dataset. An idle file is closed after 5 s and reopened when next used. Data of a new shape, or a reserve, goes to a
  hidden pending file, which is synced and then renamed over the item's file once nobody holds the old data.
- **Its first commit** adds `performance\hdf5_replace.py` to master. So that commit measures master, and the data-handle
  tip (`HEAD~1`) measures the branch. `diagnostics\windows_run.ps1` finds both commits itself.

## What is known from other machines

All timings are pinned to processors 0 and 1, medians in ms, master against the branch.

- **Two machines show the branch the same as master** for:
  - acquisition frame writes (`frame_write`);
  - loads (`load_read`);
  - rewrites at a steady shape (`rewrite`).

  One of them was measured on hard disks.
- **One machine (a Xeon w9-3495X, data on an Intel RAID volume of SSDs, Defender scanning in real time) shows the branch
  slower:**
  - `frame_write` 1.00 against 0.67;
  - `load_read` about +40%.

  On that machine, profiles found the same Python calls and the same time per function on both versions. A standalone
  probe found h5py's write, the data handle's write and the flush all equally fast. So the extra time isn't in Python
  work on the main thread, and its cause is unknown. Defender's scanning is a suspect, but untested.
- **On every machine,** operations which replace a file cost more on the branch, about 30 to 160 ms each: `reshape`,
  `reshape_held`, `reserve`, and `rewrite_after_reshape`, the first write after a replace. These are expected and
  accepted; investigate them only if step 4 asks.

## Rules

- **The repository:** don't change or commit any file in it, and don't push. Write your own scripts and notes only in
  `test-results\` (create it if needed). You may read any file in the repository, and check out the commits that
  `windows_run.ps1` uses. Return the clone to the diagnostics branch when you're done.
- **Changing behaviour for measurement:** you may monkeypatch nionswift inside your own scripts, to switch a piece off,
  but say so in your report and never present a patched result as the branch's.
- **Installing:** don't install packages or change system settings. You have no administrator rights. If something
  needed is missing, stop and say what.
- **Timing:** time only while nothing else is running on the machine. Always pin runs to processors 0 and 1, as
  `windows_run.ps1` does, and alternate master and the branch over three rounds. A single run can mislead.

## Steps

1. **Check the setup,** from the clone with the venv active:
   - `git status --short` shows no changes to tracked files;
   - `python -c "import pytest, h5py, numpy"` works;
   - nionui and nionutils, the sibling clones, are at their current masters;
   - `Get-MpComputerStatus | Format-List AMRunningMode, RealTimeProtectionEnabled` shows whether Defender is scanning.

2. **Run the standard measurement** on the laptop's SSD, with the machine quiet. It takes about 40 minutes:
   ```
   powershell -ExecutionPolicy Bypass -File .\diagnostics\windows_run.ps1 -WorkDir C:\perftemp
   ```
   Check `test-results\affinity.txt` shows affinity 3 for every python process, and that both test suites pass.

3. **Compare** each row's median between master and the branch, round by round. A row is slower on the branch only if
   all three rounds agree, beyond the spread of master's rounds.

4. **If `frame_write` or `load_read` is slower on the branch,** find which part of the branch causes it:
   - **Profile:** run `diagnostics\windows_profile.py` at both commits (`--help` isn't available; read its docstring).
     Compare call counts and time per function.
   - **Sample threads:** run `diagnostics\windows_thread_sampler.py`, which samples every thread while it runs.
   - **Switch pieces off:** write variant scripts in `test-results\` which run the branch's real path from
     `hdf5_replace.py` (its `HDF5Replace` class: `setup`, `open_write_project`, `begin_frames`, `write_frame`,
     `end_frames`, `open_load_project`, `load_read`, `teardown`). Switch off one piece per variant, by monkeypatching in
     the script. Candidates:
     - the display of the item;
     - the idle-file queue and closing (`HDF5IdleFiles`, `__queue_if_idle` in `nion\swift\model\HDF5Handler.py`);
     - the document model's `close_released_files` task;
     - the data handle's write lock;
     - the flush after a partial update.

     Run each variant pinned, alternating with the unpatched branch, three rounds. The variant which brings the branch
     back to master's time names the cause.
   - **Rule out the machine:** check whether the cause is the machine rather than the branch. Time opening, writing and
     closing a 256 MiB HDF5 file in a fresh folder against an old folder, and right after a rename against a file left
     alone, in a plain Python script with no nionswift. Defender scanning a newly created or renamed file on open would
     show there.

5. **If nothing is slower,** say so, report the numbers, and stop.

## Report

Write `test-results\REPORT.md` in plain language:
- the machine (CPU, drive, Defender state);
- the results of steps 2 and 3, as a table of medians per round for each row;
- for step 4, each experiment: what it changed, how it was run, its numbers, and what it shows;
- your conclusion about the cause, saying how sure you are and what would confirm it.

Keep your scripts in `test-results\` too. When you're done, tell the user to zip `test-results`, for example with
`Compress-Archive -Force test-results test-results-laptop.zip`, and bring the zip back.
