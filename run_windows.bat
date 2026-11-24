@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto no_launcher

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto no_python

if exist ".venv\Scripts\python.exe" goto install

echo Criando ambiente virtual...
py -3 -m venv .venv
if errorlevel 1 goto error

:install
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
python main.py
if errorlevel 1 goto error
exit /b 0

:no_launcher
echo Python Launcher py nao encontrado. Instale Python 3.11 ou superior.
goto stop

:no_python
echo Python 3.11 ou superior nao encontrado. Instale uma versao compativel.
goto stop

:error
echo.
echo Ocorreu um erro ao preparar ou executar o SSV Engine.

:stop
pause
exit /b 1
