@echo off
echo 🚀 Iniciando instalación de dependencias...

REM Actualizar pip
python -m pip install --upgrade pip

REM Instalar librerías principales
pip install pandas matplotlib ipywidgets ipython jupyter future lightkurve

REM Activar widgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix

echo ✅ Instalación completada correctamente.
pause
    