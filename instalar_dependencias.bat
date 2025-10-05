@echo off
echo 🚀 Iniciando instalación de dependencias...

REM Actualizar pip
python -m pip install --upgrade pip

REM Instalar librerías (versiones compatibles con Python 3.12)
pip install --upgrade ^
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

REM Activar widgets de Jupyter
jupyter nbextension enable --py widgetsnbextension --sys-prefix

echo ✅ Instalación completada correctamente.
pause
