# Reproducible build of the one-folder app:  powershell -ExecutionPolicy Bypass -File desktop\packaging\build.ps1
# Creates an isolated venv in desktop\packaging\.build\venv (base Python untouched), installs pinned deps, runs PyInstaller.
# Output: desktop\packaging\dist\BG Flood Viewer\BG Flood Viewer.exe   (data folder is NOT copied in)
param([string]$Python = "$env:USERPROFILE\anaconda3\python.exe", [switch]$Clean)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$venv = Join-Path $here ".build\venv"
$t0 = Get-Date
if ($Clean -and (Test-Path $venv)) { Remove-Item -Recurse -Force $venv }
if (-not (Test-Path "$venv\Scripts\python.exe")) { & $Python -m venv $venv; if ($LASTEXITCODE) { throw "venv failed" } }
$py = "$venv\Scripts\python.exe"
& $py -m pip install --disable-pip-version-check -q -r requirements-build.txt
if ($LASTEXITCODE) { throw "pip install failed" }
& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath .build\work bgv.spec
if ($LASTEXITCODE) { throw "PyInstaller failed" }
$out = Join-Path $here "dist\BG Flood Viewer"
Copy-Item QUICKSTART.md $out -Force
$mb = [math]::Round(((Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 0)
Write-Host ("Built {0}  size {1} MB  in {2:n0} s" -f $out, $mb, ((Get-Date) - $t0).TotalSeconds)
