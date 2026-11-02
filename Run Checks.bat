@echo off
rem ====================================================================
rem  DSOC Flood Map - check this installation (runs selfcheck.py)
rem
rem  Run Checks.bat                 readable table, then waits for a key
rem  Run Checks.bat --deep          also verify every flow grid's checksum
rem  Run Checks.bat /nopause        do not wait for a key at the end
rem  Any other option goes to selfcheck.py (--json, --brief, --strict, -v).
rem
rem  Exit codes: 0 no FAIL, 1 something FAILs, 11 selfcheck.py missing,
rem              12 Python not found
rem ====================================================================
setlocal EnableExtensions DisableDelayedExpansion
title DSOC Flood Map - Run Checks
set "ROOT=%~dp0"
set "RC=0"
set "PYTHON="
set "NOPAUSE="
set "ARGS="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="/nopause" set "NOPAUSE=1" & shift & goto parse
set ARGS=%ARGS% %1
shift
goto parse
:parsed

if not exist "%ROOT%selfcheck.py" goto err_missing
call :find_python
if not defined PYTHON goto err_nopython

"%PYTHON%" "%ROOT%selfcheck.py" %ARGS%
set "RC=%ERRORLEVEL%"
goto finish

:err_missing
echo [ERROR] selfcheck.py is not in this folder:
echo "%ROOT%"
echo Copy the whole "Bowling Green Flood Map" folder again.
set "RC=11"
goto finish

:err_nopython
echo [ERROR] Python 3.8 or newer was not found on this computer.
echo.
echo Install it from  https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer (Anaconda also works).
echo If Python is installed somewhere unusual, set the DSOC_PYTHON environment
echo variable to the full path of python.exe.
set "RC=12"

:finish
echo.
if not defined NOPAUSE pause
endlocal & exit /b %RC%

rem ---------------------------------------------------------------- Python search
:find_python
set "PYTHON="
if defined DSOC_PYTHON call :try "%DSOC_PYTHON%"
if defined PYTHON exit /b 0
set "PYCAND="
where py >nul 2>&1
if not errorlevel 1 for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYCAND=%%P"
if defined PYCAND call :try "%PYCAND%"
if defined PYTHON exit /b 0
call :try "%USERPROFILE%\anaconda3\python.exe"
call :try "%USERPROFILE%\miniconda3\python.exe"
call :try "%USERPROFILE%\miniforge3\python.exe"
call :try "%ProgramData%\anaconda3\python.exe"
call :try "%ProgramData%\miniconda3\python.exe"
if defined PYTHON exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python" for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do call :try "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
if defined PYTHON exit /b 0
for /f "delims=" %%P in ('where python 2^>nul') do call :try "%%P"
exit /b 0

:try
if defined PYTHON exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 3)" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON=%~1"
exit /b 0
