@echo off
REM Script de inicio para Pokedex Animal Ultra - Windows Edition
REM Configura variables de entorno y ejecuta la aplicacion

echo ============================================================
echo    POKEDEX ANIMAL ULTRA - WINDOWS PROFESSIONAL EDITION
echo ============================================================
echo.

REM Suprimir mensajes informativos de TensorFlow
set TF_CPP_MIN_LOG_LEVEL=1

REM Desactivar oneDNN si se requiere determinismo (comentar para mejor rendimiento)
REM set TF_ENABLE_ONEDNN_OPTS=0

echo Iniciando aplicacion...
echo.

REM Ejecutar aplicacion
python pokedex_ultra_windows.py

echo.
echo Aplicacion finalizada.
pause
