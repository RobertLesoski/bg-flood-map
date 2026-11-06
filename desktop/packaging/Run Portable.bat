@echo off
rem Portable route: creates (once) and uses a conda env "bgv" from environment.yml, then runs the viewer from source.
rem Data folder: same rules as the exe (bgv_data.txt next to this file, BGV_DATA_ROOT, .\data, default path).
setlocal
set "HERE=%~dp0"
where conda >nul 2>nul || if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "PATH=%USERPROFILE%\anaconda3\condabin;%PATH%"
where conda >nul 2>nul || (echo Conda not found. Install Miniconda or Anaconda first. & pause & exit /b 1)
call conda run -n bgv python -c "import wgpu" >nul 2>nul || call conda env create -f "%HERE%environment.yml" || (pause & exit /b 1)
set "PYTHONPATH=%HERE%.."
call conda run --no-capture-output -n bgv python "%HERE%bgv_entry.py" %*
