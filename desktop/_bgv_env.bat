@echo off
rem Helper for the BG Flood desktop viewer launchers: sets PY (python.exe) and PYW (pythonw.exe, no console window).
rem   BGV_PYTHON=<full path to python.exe>   forces a specific interpreter
rem Order: BGV_PYTHON, Anaconda / Miniconda in the usual places, the "py -3" launcher, python on PATH.
rem (BGV_FORCE_NO_PYTHON=1 simulates a machine without Python, for tests.)
rem Returns errorlevel 1 (and leaves PY empty) when no working Python 3 was found.
set "PY="
set "PYW="
if "%BGV_FORCE_NO_PYTHON%"=="1" exit /b 1
if defined BGV_PYTHON if exist "%BGV_PYTHON%" set "PY=%BGV_PYTHON%"
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%LOCALAPPDATA%\anaconda3\python.exe" "%LOCALAPPDATA%\miniconda3\python.exe" "%ProgramData%\anaconda3\python.exe" "%ProgramData%\miniconda3\python.exe" "C:\anaconda3\python.exe" "C:\miniconda3\python.exe") do (
  if not defined PY if exist %%P set "PY=%%~P"
)
if not defined PY (
  where py >nul 2>nul && for /f "usebackq delims=" %%P in (`py -3 -c "import sys;print(sys.executable)" 2^>nul`) do (
    if not defined PY if exist "%%P" set "PY=%%P"
  )
)
if not defined PY (
  where python >nul 2>nul && for /f "usebackq delims=" %%P in (`where python`) do (
    if not defined PY "%%P" -c "import sys" >nul 2>nul && set "PY=%%P"
  )
)
if not defined PY exit /b 1
for %%P in ("%PY%") do set "PYW=%%~dpPpythonw.exe"
if not exist "%PYW%" set "PYW=%PY%"
exit /b 0
