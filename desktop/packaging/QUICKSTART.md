# BG Flood Viewer - presenter quickstart

**Start:** double-click `BG Flood Viewer.exe` (in the `BG Flood Viewer` folder). For the county view straight away,
make a shortcut with target `"...\BG Flood Viewer.exe" --domain county` (the last area is also remembered).
First start of a new install builds caches (about 1-2 min); later starts are fast. Do a dry run before presenting.

**Data folder** (about 7 GB, not inside the app; must contain `layers\flow`). First match wins:
1. `--data "<folder>"` on the command line
2. a text file `bgv_data.txt` next to the exe containing the folder path
3. environment variable `BGV_DATA_ROOT` (or `BGV_DATA`)
4. a folder named `data` next to the exe
5. default `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map`

If none is found, a message box says so.

**Keys:** F1 controls help - H hide/show interface - F12 screenshot - Ctrl+F address search -
W/A/S/D move, mouse drag pan, wheel zoom - Space play/pause (see F1 for the full list).

**Logs:** `bgv.log` next to the exe; on a crash `bgv_error.log` plus a message box.

**Known issues**
- Blank text (no UI labels) was seen once on a second monitor at 125% scaling; **not confirmed fixed**.
  Present on the primary monitor, or set that display to 100%, and test beforehand.
- Needs a Vulkan/DX12 GPU (tested: GTX 1650, ~46 fps county view).
- Windows SmartScreen may warn on an unsigned exe: "More info" -> "Run anyway".
- Fallback if the exe will not start: `Run Portable.bat` (needs conda; builds env `bgv` once).
