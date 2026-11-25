# Run the data-handle branch's tests, performance script and probes on Windows, and save the results as UTF-8 text.
#
# Usage, from the nionswift clone with the diagnostics branch checked out, the venv active (nionswift's dependencies and
# pytest), a clean working tree, and nionui and nionutils at their current masters:
#     powershell -ExecutionPolicy Bypass -File .\diagnostics\windows_run.ps1 -WorkDir <folder on the drive to measure>
# Results go into .\test-results. Keep the machine quiet during the timing rounds, which run on processors 0 and 1 only.
#
# The diagnostics branch is the data-handle branch plus one commit holding this folder. The data-handle branch's first
# commit adds performance\hdf5_replace.py to master, so that commit measures master and the data-handle tip measures the
# branch. The scripts are copied out of the checkout first, since this folder exists only at the diagnostics commit.
#
# Every file the runs write goes into WorkDir: the timing projects and the probes' files through TEMP and TMP, and the test
# suites' __Test folders by running them from it. work-drive.txt records its drive and the kind of disk.

param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [switch]$SkipTests
)

# native commands write progress to stderr, which must not stop the script.
$ErrorActionPreference = "Continue"
$clone = (Get-Location).Path
$results = Join-Path $clone "test-results"
# the clone's nion comes first, so that each run measures the checked out commit and not an installed nionswift.
$env:PYTHONPATH = $clone

# stop before anything long if a needed package is missing or the checkout isn't the diagnostics branch.
python -c "import pytest, h5py, numpy" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "pytest, h5py or numpy is missing from the active Python; install them in the venv and run again."
    exit 1
}
$diagnosticsBranch = (git branch --show-current).Trim()
if (-not (Test-Path (Join-Path $clone "diagnostics\windows_run.ps1"))) {
    Write-Output "Run this from the nionswift clone with the diagnostics branch checked out."
    exit 1
}
if (git status --short --untracked-files=no) {
    Write-Output "The working tree has changes; commit or stash them first."
    exit 1
}
$branchTip = (git rev-parse "HEAD~1").Trim()
$first = (git log --diff-filter=A --format=%H $branchTip -- performance/hdf5_replace.py | Select-Object -Last 1).Trim()
Write-Output "diagnostics at $(git rev-parse --short HEAD), data-handle tip $($branchTip.Substring(0, 8)), master measured at $($first.Substring(0, 8))"

New-Item -ItemType Directory -Force -Path $results | Out-Null
$tools = Join-Path $results "tools"
New-Item -ItemType Directory -Force -Path $tools | Out-Null
Copy-Item (Join-Path $clone "diagnostics\*.py") $tools -Force
Copy-Item (Join-Path $clone "performance\hdf5_replace.py") $tools -Force
$script = Join-Path $tools "hdf5_replace.py"

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$WorkDir = (Resolve-Path $WorkDir).Path
# python's tempfile takes its folder from TEMP and TMP, so the timing projects and the probes' files go into WorkDir.
$env:TEMP = $WorkDir
$env:TMP = $WorkDir
$driveLetter = $WorkDir.Substring(0, 1)
$disk = Get-Partition -DriveLetter $driveLetter -ErrorAction SilentlyContinue | Get-Disk -ErrorAction SilentlyContinue
$driveText = @(
    "work folder $WorkDir",
    "drive $driveLetter, disk $($disk.Number) $($disk.FriendlyName), bus $($disk.BusType)",
    ((Get-PhysicalDisk -ErrorAction SilentlyContinue | Where-Object { $_.DeviceId -eq "$($disk.Number)" }) | ForEach-Object { "media $($_.MediaType)" }),
    ((Get-MpComputerStatus -ErrorAction SilentlyContinue) | ForEach-Object { "defender $($_.AMRunningMode), real-time protection $($_.RealTimeProtectionEnabled)" })
)
$driveText | Out-File -Encoding utf8 (Join-Path $results "work-drive.txt")
$driveText
$affinityFile = Join-Path $results "affinity.txt"
Remove-Item $affinityFile -ErrorAction SilentlyContinue

function Save($name, [scriptblock]$command) {
    $file = Join-Path $results $name
    & $command 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 $file
    Get-Content $file -Tail 3
}

# Run python with arguments on processors 0 and 1 only, so that Windows' placement of the process doesn't decide its times,
# and save its output as UTF-8 in the file name, followed by its errors.
function SavePinned($name, [string[]]$arguments) {
    $file = Join-Path $results $name
    $errorFile = Join-Path $results "$name.errors"
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    # python inherits this shell's affinity when it starts, and passes it on to an interpreter which it starts in turn, as
    # a venv's python.exe does; an affinity set on the started process afterwards doesn't reach that interpreter. The
    # started process's affinity is set as well, in case this shell's can't be set.
    $shell = [System.Diagnostics.Process]::GetCurrentProcess()
    $shellAffinity = $shell.ProcessorAffinity
    $shell.ProcessorAffinity = 0x3
    # Start-Process gives python the file itself as its output, so the file holds python's own bytes, which are in the
    # ANSI code page unless PYTHONIOENCODING says otherwise.
    $env:PYTHONIOENCODING = "utf-8"
    $process = Start-Process python -ArgumentList $arguments -RedirectStandardOutput $file -RedirectStandardError $errorFile -NoNewWindow -PassThru
    Remove-Item Env:PYTHONIOENCODING
    $process.ProcessorAffinity = 0x3
    $shell.ProcessorAffinity = $shellAffinity
    # record the affinity of each python process once the interpreter has started, which shows whether the pinning held.
    Start-Sleep -Seconds 3
    $affinities = Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { "process $($_.Id) affinity $($_.ProcessorAffinity)" }
    [System.IO.File]::AppendAllText($affinityFile, "${name}: $($affinities -join ', ')`r`n", $utf8)
    $process.WaitForExit()
    $errorText = [System.IO.File]::ReadAllText($errorFile, $utf8)
    if ($errorText) {
        [System.IO.File]::AppendAllText($file, $errorText, $utf8)
    }
    Remove-Item $errorFile
    Get-Content $file -Tail 3 -Encoding UTF8
}

for ($round = 1; $round -le 3; $round++) {
    git switch --quiet --detach $first
    SavePinned "timing-round$round-master.txt" @("`"$script`"", "--label", "master")
    git switch --quiet --detach $branchTip
    SavePinned "timing-round$round-branch.txt" @("`"$script`"", "--label", "branch")
}

git switch --quiet --detach $branchTip
Save "frame-probe.txt" { python (Join-Path $tools "windows_frame_probe.py") }
Save "lock-probe.txt" { python (Join-Path $tools "windows_lock_probe.py") }

if (-not $SkipTests) {
    $tests = Join-Path $clone "nion\swift\test"
    # the suites run from WorkDir, where their file tests make their __Test folders.
    git switch --quiet --detach $first
    Push-Location $WorkDir
    Save "tests-master.txt" { python -m pytest -p no:cacheprovider -q -rfE $tests }
    Pop-Location
    git switch --quiet --detach $branchTip
    Push-Location $WorkDir
    Save "tests-dh.txt" { python -m pytest -p no:cacheprovider -q -rfE $tests }
    Pop-Location
}

git switch --quiet $diagnosticsBranch
Write-Output "results in $results"
