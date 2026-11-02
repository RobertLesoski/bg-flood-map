@echo off
rem Runs the desktop viewer checks and prints PASS or FAIL:
rem   1. the pytest suite (unit tests + offscreen GPU tests, no window)
rem   2. a headless smoke render (python -m bgv --render, no window)
rem Options: --tests-only, --smoke-only.  (Test hooks: BGV_PYTEST_ARGS replaces the pytest target, BGV_SMOKE_ARGS adds render arguments.)   Exit code 0 = everything passed.
setlocal EnableExtensions
cd /d "%~dp0"
set "DO_TESTS=1"
set "DO_SMOKE=1"
if /i "%~1"=="--tests-only" set "DO_SMOKE="
if /i "%~1"=="--smoke-only" set "DO_TESTS="
set "RC=0"
set "T=skipped"
set "S=skipped"

call "%~dp0_bgv_env.bat"
if not defined PY goto nopython
echo Using %PY%

if not defined DO_TESTS goto smoke
echo.
echo === 1/2  pytest suite ===
set "PT=tests"
if defined BGV_PYTEST_ARGS set "PT=%BGV_PYTEST_ARGS%"
"%PY%" -m pytest %PT% -q -p no:cacheprovider
if errorlevel 1 (set "T=FAIL") else set "T=PASS"
if "%T%"=="FAIL" set "RC=1"

:smoke
if not defined DO_SMOKE goto summary
echo.
echo === 2/2  headless smoke render ===
set "SMOKE=%TEMP%\bgv_smoke.png"
del "%SMOKE%" >nul 2>nul
"%PY%" -m bgv --render "%SMOKE%" --storm 25yr --hour 12 --size 640x360 --quiet %BGV_SMOKE_ARGS%
set "S=FAIL"
if exist "%SMOKE%" for %%F in ("%SMOKE%") do if %%~zF GTR 5000 set "S=PASS"
if not "%S%"=="PASS" set "RC=1"
if "%S%"=="PASS" echo Smoke image: %SMOKE%

:summary
echo.
echo ============================================
echo   pytest suite ...... %T%
echo   smoke render ...... %S%
if "%RC%"=="0" (echo   OVERALL ........... PASS) else (echo   OVERALL ........... FAIL)
echo ============================================
goto done

:nopython
echo Python 3 was not found. Install Anaconda from https://www.anaconda.com or set BGV_PYTHON to the full path of python.exe.
set "RC=1"

:done
rem started by a double click: keep the window open so the result can be read
echo %cmdcmdline%| findstr /i /c:"/c" >nul && if not defined BGV_LAUNCH_MODE pause
exit /b %RC%
