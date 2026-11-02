@echo off
rem Debug launcher: same as "Launch Desktop Viewer.bat --console": the viewer runs in this window, messages stay visible.
call "%~dp0Launch Desktop Viewer.bat" --console %*
exit /b %errorlevel%
