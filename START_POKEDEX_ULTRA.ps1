# Script de inicio para Pokedex Animal Ultra - Windows Edition
# Configura variables de entorno y ejecuta la aplicacion

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   POKEDEX ANIMAL ULTRA - WINDOWS PROFESSIONAL EDITION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Suprimir mensajes informativos de TensorFlow
$env:TF_CPP_MIN_LOG_LEVEL = "1"

# Desactivar oneDNN si se requiere determinismo (comentar para mejor rendimiento)
# $env:TF_ENABLE_ONEDNN_OPTS = "0"

Write-Host "Iniciando aplicacion..." -ForegroundColor Green
Write-Host ""

# Ejecutar aplicacion
python pokedex_ultra_windows.py

Write-Host ""
Write-Host "Aplicacion finalizada." -ForegroundColor Yellow
Read-Host "Presiona Enter para salir"
