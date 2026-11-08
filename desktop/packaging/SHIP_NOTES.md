# BG Flood Viewer: ship notes

- **Version:** git commit 24735ae (desktop-integration, wrap-up 2026-11-08). The exe was built from this commit.
- **Folder:** `desktop\packaging\dist\BG Flood Viewer\` (500 MB, one folder, start `BG Flood Viewer.exe`).
  Zip of the same folder: `desktop\packaging\dist\BG Flood Viewer.zip` (362 MB). The 7 GB data folder is NOT included.
- **Build:** `powershell -ExecutionPolicy Bypass -File desktop\packaging\build.ps1` (isolated venv, PyInstaller), 133 s.

## What is in it
The native wgpu/Vulkan viewer: city and Warren County domains, all design and historical storms, 24 h playback,
solver-depth overlay, buildings / roads at risk, address search, utilities, flood tools (route, exposure, duration,
compare, share), cinematic tour / time-lapse / export, analysis tools, help with the keyboard cheat sheet. QUICKSTART.md
is copied next to the exe.

## How to run
1. Copy the folder (or unzip) anywhere. Needs Windows 10/11 64-bit and a GPU with Vulkan or DX12.
2. Tell it where the data is (folder that contains `layers\flow`), one of:
   - `--data "D:\Bowling Green Flood Map"` on the command line, or
   - a text file `bgv_data.txt` next to the exe containing that folder path (one line), or
   - environment variable `BGV_DATA_ROOT` (or `BGV_DATA`).
3. Double-click `BG Flood Viewer.exe`. Checks: `"BG Flood Viewer.exe" --selftest` (3 s window, exits) and
   `--render out.png --ui` (offscreen frame).

## Smoke test on the build PC (GTX 1650, real data)
- `--domain county --render ... --ui --size 1600x900`: OK, 73 s (cold start in the packaged app, caches built).
- `--domain county --selftest`: OK, 45 fps, mouse drag / wheel / W key reached the camera, no errors, 14.9 s.

## Known issues
- Blank interface text on a SECOND monitor at 125 % scale was reported; a fix went in earlier but it is NOT confirmed
  on that setup (one monitor here).
- The first start on a new PC builds caches: about 1-2 minutes before the first frame (later starts < 15 s).
- The exe is unsigned: Windows SmartScreen may warn ("More info" -> "Run anyway").
- Storms 10-12 have no number key (use the sidebar or the Storm menu).
