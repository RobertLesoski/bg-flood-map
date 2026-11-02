@echo off
rem ====================================================================
rem  DSOC Bowling Green Flood Risk Map - launcher
rem
rem  Double-click for the menu, or pass options (no menu):
rem    Launch Map.bat 1            open the 3D Flood Risk Map
rem    Launch Map.bat 2            open the Flood Probability Report
rem    Launch Map.bat 3            open both
rem    Launch Map.bat 4            check this installation (no server)
rem    /nobrowser                  start the server but do not open a browser
rem    /port N                     preferred port (default 8000; the next
rem                                free one is used if it is taken)
rem    /nopause                    never wait for a key (see below)
rem  The environment variable DSOC_PORT changes the default port.
rem
rem  After an error, and after option 4, the window waits for a key, so that a
rem  shortcut or a double-click does not close before the message can be read.
rem  Scripts that call this file add /nopause.
rem
rem  Exit codes: 0 ok, 10 bad option, 11 folder incomplete,
rem              12 Python not found, 13 server did not start,
rem              14 self-check found problems (option 4)
rem  Python can be forced with the DSOC_PYTHON environment variable.
rem  The menu (no options) needs a keyboard; scripts should pass 1-4.
rem ====================================================================
setlocal EnableExtensions DisableDelayedExpansion
title DSOC - Bowling Green Flood Risk Map
set "ROOT=%~dp0"
set "RC=0"
set "CHOICE="
set "NOBROWSER="
set "NOPAUSE="
set "PORT=8000"
set "INTERACTIVE=0"
if not defined DSOC_PORT goto port_default_done
set "PORT=%DSOC_PORT%"
call :port_ok
if errorlevel 1 echo   Ignoring DSOC_PORT, it must be a number from 1024 to 65535.
if errorlevel 1 set "PORT=8000"
:port_default_done
set "PYTHON="
set "BASE="

:parse
if "%~1"=="" goto parsed
set "ARG=%~1"
if /i "%ARG%"=="/?" goto usage
if /i "%ARG%"=="-h" goto usage
if /i "%ARG%"=="--help" goto usage
if /i "%ARG%"=="/help" goto usage
if /i "%ARG%"=="/nobrowser" set "NOBROWSER=1" & shift & goto parse
if /i "%ARG%"=="-nobrowser" set "NOBROWSER=1" & shift & goto parse
if /i "%ARG%"=="--no-browser" set "NOBROWSER=1" & shift & goto parse
if /i "%ARG%"=="/nopause" set "NOPAUSE=1" & shift & goto parse
if /i "%ARG%"=="-nopause" set "NOPAUSE=1" & shift & goto parse
if /i "%ARG%"=="--no-pause" set "NOPAUSE=1" & shift & goto parse
if /i "%ARG%"=="/port" goto parse_port
if "%ARG%"=="1" set "CHOICE=1" & shift & goto parse
if "%ARG%"=="2" set "CHOICE=2" & shift & goto parse
if "%ARG%"=="3" set "CHOICE=3" & shift & goto parse
if "%ARG%"=="4" set "CHOICE=4" & shift & goto parse
echo Unknown option: "%ARG%"
set "RC=10"
goto bad_args

:parse_port
shift
set "PORT=%~1"
if "%PORT%"=="" goto bad_port
call :port_ok
if errorlevel 1 goto bad_port
shift
goto parse

:bad_port
echo The port after /port must be a number from 1024 to 65535.
set "RC=10"

:bad_args
rem  parsing stops at the mistake, but a /nopause after it still counts
shift
:scan_nopause
if "%~1"=="" goto usage_bad
if /i "%~1"=="/nopause" set "NOPAUSE=1"
shift
goto scan_nopause

:parsed
if not defined CHOICE if not defined NOBROWSER set "INTERACTIVE=1"

color 04
echo.
echo ================================================================
echo   DISASTER SCIENCE OPERATIONS CENTER (DSOC)
echo   AI Research Lab (AIR)
echo   Western Kentucky University
echo ================================================================
echo   Bowling Green, KY -- 3D Flood Risk ^& Property Map
echo   Made by: Robert Lesoski
echo ================================================================
echo.

rem ---- the folder must be complete enough to run ----
if "%CHOICE%"=="4" goto need_selfcheck
if not exist "%ROOT%serve.py" goto err_incomplete
if not exist "%ROOT%viewer.html" goto err_incomplete
goto have_files
:need_selfcheck
if not exist "%ROOT%selfcheck.py" goto err_incomplete
:have_files

rem ---- Python ----
call :find_python
if not defined PYTHON goto err_nopython

if "%CHOICE%"=="4" goto selfcheck_only

echo   Checking that the map data are in place...
if not exist "%ROOT%selfcheck.py" goto sc_done
rem  --summary prints one line, or the problem rows when something FAILs; no temp file (two launchers at once would share its name)
"%PYTHON%" "%ROOT%selfcheck.py" --summary
if "%ERRORLEVEL%"=="0" goto sc_ok
echo.
echo   Some problems were found (see above). The map will still open, but parts
echo   of it may be missing. Run "Run Checks.bat" for the full list.
goto sc_done
:sc_ok
echo   (details: Run Checks.bat)
:sc_done
echo.

rem ---- start (or reuse) the server ----
echo   Starting the local map server...
call :find_server
if defined BASE goto have_server
"%PYTHON%" "%ROOT%serve.py" --spawn --port %PORT%
if errorlevel 1 goto err_server
call :find_server --wait 10
if not defined BASE goto err_server
:have_server
echo.
echo   Server running at %BASE%
echo.

if defined CHOICE goto act
if defined NOBROWSER goto done_nobrowser

:menu
echo   [1]  3D Flood Risk Map
echo   [2]  Flood Probability Report ^& TXT Export
echo   [3]  Open Both
echo   [4]  Check this installation (self-check)
echo.
set "CHOICE="
set /p "CHOICE=  Enter choice (1/2/3/4): "
rem an empty answer leaves CHOICE undefined, and substituting into an undefined variable breaks the parser
if not defined CHOICE goto menu_invalid
set "CHOICE=%CHOICE:"=%"
if not defined CHOICE goto menu_invalid
if "%CHOICE%"=="4" goto menu_check
if "%CHOICE%"=="1" goto act
if "%CHOICE%"=="2" goto act
if "%CHOICE%"=="3" goto act
:menu_invalid
set /a BAD+=1
if %BAD% GEQ 5 goto done_nochoice
echo   Please type 1, 2, 3 or 4.
echo.
goto menu

:menu_check
echo.
"%PYTHON%" "%ROOT%selfcheck.py"
echo.
goto menu

:act
if defined NOBROWSER goto done_nobrowser
if "%CHOICE%"=="1" start "" "%BASE%/viewer.html"
if "%CHOICE%"=="2" goto open_report_only
if "%CHOICE%"=="3" goto open_report_too
goto opened
:open_report_only
if not exist "%ROOT%flood_report.html" echo   flood_report.html is not in this folder.
if exist "%ROOT%flood_report.html" start "" "%BASE%/flood_report.html"
goto opened
:open_report_too
if not exist "%ROOT%flood_report.html" echo   flood_report.html is not in this folder.
if exist "%ROOT%flood_report.html" start "" "%BASE%/flood_report.html"
start "" "%BASE%/viewer.html"
:opened
echo.
echo   Keep the "DSOC Flood Map Server" window open while using the map.
echo   Close that window to stop the server.
echo.
if "%INTERACTIVE%"=="1" if not defined NOPAUSE echo   Press any key to close this window. The server keeps running.
if "%INTERACTIVE%"=="1" if not defined NOPAUSE pause >nul
goto finish

:done_nobrowser
echo   Open  %BASE%/viewer.html  in your browser.
echo   Report: %BASE%/flood_report.html
echo   (server window: "DSOC Flood Map Server" - close it to stop the server)
goto finish

:done_nochoice
echo   No choice made. The server keeps running at %BASE%
goto finish

:selfcheck_only
"%PYTHON%" "%ROOT%selfcheck.py"
if errorlevel 1 set "RC=14"
echo.
call :wait_key
goto finish

:finish
endlocal & exit /b %RC%

rem ---------------------------------------------------------------- errors
:err_incomplete
echo   [ERROR] This folder looks incomplete: serve.py and viewer.html must be here.
echo   Folder: "%ROOT%"
echo   Copy the whole "Bowling Green Flood Map" folder again, then run this file.
echo   (Run Checks.bat lists exactly what is missing.)
set "RC=11"
goto fail

:err_nopython
echo   [ERROR] Python 3.8 or newer was not found on this computer.
echo.
echo   Install it from  https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" in the installer (Anaconda also works).
echo   Then double-click this file again.
echo   If Python is installed somewhere unusual, set the DSOC_PYTHON environment
echo   variable to the full path of python.exe.
set "RC=12"
goto fail

:err_server
echo   [ERROR] The map server did not start within 10 seconds.
echo   Look at the "DSOC Flood Map Server" window for the reason (it stays open
echo   after an error). Try another port:  Launch Map.bat /port 8200
echo   Python used: "%PYTHON%"
set "RC=13"
goto fail

:usage
rem  asking for help is not an error: nothing to read before the window closes
set "RC=0"
set "NOPAUSE=1"
:usage_bad
echo.
echo Usage: Launch Map.bat [1^|2^|3^|4] [/nobrowser] [/port N]
echo   1   open the 3D Flood Risk Map
echo   2   open the Flood Probability Report
echo   3   open both
echo   4   check this installation (self-check, no server)
echo   /nobrowser   start the server but do not open a browser
echo   /port N      preferred port, 1024-65535 (default 8000)
echo   With no options a menu is shown.
goto fail_noeko

:fail
echo.
:fail_noeko
call :wait_key
endlocal & exit /b %RC%

rem ---------------------------------------------------------------- wait for a key
rem  Keeps the window readable after an error or the self-check; /nopause skips it.
rem  The line is printed here because the text of PAUSE depends on the Windows language.
:wait_key
if defined NOPAUSE exit /b 0
echo   Press any key to close this window.
pause >nul
exit /b 0

rem ---------------------------------------------------------------- server lookup
rem  Sets BASE to the address of the server already running for this folder, or leaves it empty.
rem  Arguments (--wait 10) go to  serve.py --find.  The address is read straight from the pipe: a temp file named
rem  with %RANDOM% collided when two launchers started within a second.
:find_server
set "BASE="
rem  The command starts with @ on purpose: cmd /c drops the first and last quote of a command that starts with a quote,
rem  and that breaks as soon as a path has a space (tested with spaces, parentheses and & in the folder and in the python path).
for /f "usebackq delims=" %%U in (`@"%PYTHON%" "%ROOT%serve.py" --find %* --port %PORT% 2^>nul`) do set "BASE=%%U"
exit /b 0

rem ---------------------------------------------------------------- port check
rem  errorlevel 0 when %PORT% is a whole number from 1024 to 65535.
rem  "set PORT" lists the variable instead of pasting its text into a command line, so odd characters are harmless.
:port_ok
set PORT| findstr /r /x "PORT=[0-9][0-9]*" >nul
if errorlevel 1 exit /b 1
if %PORT% LSS 1024 exit /b 1
if %PORT% GTR 65535 exit /b 1
exit /b 0

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
