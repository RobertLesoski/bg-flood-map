@echo off
rem BG Flood desktop viewer launcher.
rem   Launch Desktop Viewer.bat [viewer arguments]      starts the viewer with pythonw: no console window stays open
rem   Launch Desktop Viewer.bat --console [arguments]   runs in this console (messages, tracebacks); also: "Launch Desktop Viewer (console).bat"
rem Arguments are passed on to "python -m bgv" (--data <folder>, --storm 25yr, --size 1600x900, --debug ...).
rem Started from a console the batch waits for the viewer and returns its exit code; started by a double click it
rem returns at once.  Missing Python / missing packages: a message box with the exact pip command.
rem Test hooks: BGV_LAUNCH_DRYRUN=1 prints what would run; BGV_LAUNCH_MODE=wait|detach forces the wait behaviour;
rem BGV_PKGS=a,b overrides the list of required Python modules; BGV_NO_DIALOG=1 prints instead of showing boxes.
setlocal EnableExtensions
rem the folder must be captured first: %~dp0 follows %0 and changes after SHIFT
set "HERE=%~dp0"
cd /d "%HERE%"
set "CONSOLE="
set "ARGS="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--console" (set "CONSOLE=1") else set "ARGS=%ARGS% "%~1""
shift
goto parse
:parsed

call "%HERE%_bgv_env.bat"
if not defined PY goto nopython

set "PKGS=wgpu,rendercanvas,glfw,imgui_bundle,numpy,scipy,numba,rasterio,shapely,PIL"
if defined BGV_PKGS set "PKGS=%BGV_PKGS%"
set "MISSING="
"%PY%" -c "import importlib.util as u,sys;print(','.join(n for n in sys.argv[1].split(',') if u.find_spec(n) is None))" "%PKGS%" > "%TEMP%\bgv_missing_%RANDOM%.txt" 2>nul
for %%F in ("%TEMP%\bgv_missing_*.txt") do (
  set /p MISSING=<"%%F"
  del "%%F" >nul 2>nul
)
if defined MISSING goto nopackages

rem started by a double click (cmd /c ...) or from a console?  (no parenthesised blocks around %cmdcmdline%: it may contain "(")
set "FROM_EXPLORER=1"
echo %cmdcmdline%| findstr /i /c:"/c" >nul
if errorlevel 1 set "FROM_EXPLORER="
set "MODE=detach"
if not defined FROM_EXPLORER set "MODE=wait"
if defined BGV_LAUNCH_MODE set "MODE=%BGV_LAUNCH_MODE%"
if defined CONSOLE set "MODE=console"

if defined BGV_LAUNCH_DRYRUN goto dryrun
if /i "%MODE%"=="console" goto console
if /i "%MODE%"=="wait" goto waitrun
rem double click: start and return (pythonw has no console; errors appear as a message box and in logs\bgv.log)
start "" "%PYW%" -m bgv %ARGS%
exit /b 0

:dryrun
echo [dryrun] mode=%MODE% python=%PY% pythonw=%PYW% args=%ARGS%
exit /b 0

:waitrun
start "" /wait "%PYW%" -m bgv %ARGS%
exit /b %errorlevel%

:console
"%PY%" -m bgv %ARGS%
set "RC=%errorlevel%"
if "%RC%"=="0" exit /b 0
echo.
echo The viewer stopped with exit code %RC%. Details: %HERE%logsgv.log
if defined FROM_EXPLORER if not defined BGV_LAUNCH_MODE pause
exit /b %RC%

:nopython
set "BGV_MSG=BG Flood Viewer cannot start: Python 3 was not found.| |Install Anaconda from https://www.anaconda.com (or Python 3 from https://www.python.org), then install the packages:|python -m pip install wgpu rendercanvas glfw imgui-bundle numpy numba scipy rasterio shapely mapbox_earcut pillow requests| |Or set BGV_PYTHON to the full path of python.exe."
call :msgbox
exit /b 9009

:nopackages
set "BGV_MSG=BG Flood Viewer cannot start: these Python packages are missing: %MISSING%| |Install them with this command (copy it into a command prompt):|"%PY%" -m pip install wgpu rendercanvas glfw imgui-bundle numpy numba scipy rasterio shapely mapbox_earcut pillow requests"
call :msgbox
exit /b 9009

:msgbox
rem BGV_MSG uses | as the line separator.  Message box through PowerShell (always present on Windows 10 / 11).
if "%BGV_NO_DIALOG%"=="1" goto msgconsole
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show(($env:BGV_MSG -replace '\|', [Environment]::NewLine), 'BG Flood Viewer', 'OK', 'Error')" >nul 2>nul
if not errorlevel 1 exit /b 0
:msgconsole
echo.
echo %BGV_MSG:|=&echo %
echo.
exit /b 0
