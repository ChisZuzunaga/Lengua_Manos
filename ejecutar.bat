@echo off
REM Activar el entorno de Conda
call conda activate sign_language_env

REM Ejecutar el script main.py
python src/main.py

REM Pausar para ver la salida
pause