@echo off
setlocal ENABLEDELAYEDEXPANSION

set ENV_NAME=seasonal-rings

where conda >NUL 2>&1
if %ERRORLEVEL%==0 (
  echo [setup] Using Conda environment "%ENV_NAME%" from environment.yml
  conda env update -n %ENV_NAME% -f environment.yml || conda env create -n %ENV_NAME% -f environment.yml
  echo.
  echo [setup] Done. Activate with:
  echo   conda activate %ENV_NAME%
  goto :eof
)

where mamba >NUL 2>&1
if %ERRORLEVEL%==0 (
  echo [setup] Using Mamba environment "%ENV_NAME%" from environment.yml
  mamba env update -n %ENV_NAME% -f environment.yml || mamba env create -n %ENV_NAME% -f environment.yml
  echo.
  echo [setup] Done. Activate with:
  echo   conda activate %ENV_NAME%
  goto :eof
)

echo [setup] Conda/Mamba not found. Falling back to a local virtualenv: .venv
py -m venv .venv
call .\.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install numpy pandas matplotlib
echo.
echo [setup] Done. Activate with:
echo   .\.venv\Scripts\activate
