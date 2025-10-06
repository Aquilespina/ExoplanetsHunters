@echo off
echo 🚀 Iniciando instalación de dependencias...

REM Actualizar pip
python -m pip install --upgrade pip

REM Instalar librerías principales (compatibles con Python 3.12)
pip install ^
    numpy ^
    pandas ^
    matplotlib ^
    seaborn ^
    scikit-learn>=1.4,<1.6 ^
    xgboost ^
    lightkurve ^
    ipywidgets ^
    ipython ^
    jupyter ^
    future ^
    streamlit ^
    plotly

REM Instalar dependencias adicionales desde requirements.txt (si existe)
if exist requirements.txt (
    echo 📦 Instalando dependencias adicionales desde requirements.txt...
    pip install -r requirements.txt
)

REM Activar widgets de Jupyter
jupyter nbextension enable --py widgetsnbextension --sys-prefix

echo ✅ Instalación completada correctamente.
pause
