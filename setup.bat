@echo off
call conda info >nul 2>&1 || (echo Conda not found in PATH & exit /b 1)

REM Create env from YAML
conda env list | findstr /R "^quantum-poetry " >nul
IF %ERRORLEVEL% NEQ 0 (
  echo Updating existing env "quantum-poetry"...
  conda env update -f environment.yml --prune
) ELSE (
  echo Creating env "quantum-poetry"...
  conda env create -f environment.yml
)

call conda activate quantum-poetry

python -V
python -c "import numpy, pandas, scipy, networkx, matplotlib, plotly, skimage, PIL; print('Deps OK')"

python - <<PY
import numpy, pandas, scipy, networkx, matplotlib, plotly, skimage, PIL, flask, dash
print("Dependencies imported successfully.")
PY


IF EXIST poem-v1.txt (
  echo Running quantum_poetry.py ...
  python quantum_poetry.py
) ELSE (
  echo NOTE: Put poem-v1.txt in this folder (or change input_file in the script), then run:
  echo conda activate quantum-poetry && python quantum_poetry.py
)

echo Env ready. Run with: conda activate quantum-poetry && python <your_script>.py
echo NOTE: Place your poem file at ./poem-v1.txt (or update input_file in the script) then run


