# Exoplanets Hunters
Herramienta interactiva para **detección y análisis de candidatos a exoplanetas**.  
Incluye un **dashboard en Streamlit** para cargar datos, obtener **predicciones** mediante un **modelo de aprendizaje profundo** (p. ej. CNN/Transformer/MLP) o un baseline clásico (XGBoost), y visualizar resultados con **Plotly**. Además, calcula **inferencias físicas heurísticas** (temperatura de equilibrio, flujo estelar relativo, radio estimado y clase composicional + minerales/nubes probables).

> **Nota de modelo**: El proyecto está preparado para usar un modelo DL **pre-entrenado** (PyTorch o TensorFlow) exportado en `./artifacts/`. También funciona con un modelo clásico (XGBoost) si lo prefieres.
<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/1c25d3e7-9904-4693-9bdc-108d21db6044" />
<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/b744b247-1ccd-4dfd-9b3f-8c71e372915a" />
<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/20ed789c-cf83-47dd-939f-36b1d7b6982d" />
<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/d7d4c62e-f59f-42f4-959d-ec9ca59233c6" />

---


## Requisitos
- **Python 3.10 – 3.12**  
- Librerías principales: `numpy`, `pandas`, `plotly`, `streamlit`, `joblib`.  
- **Backend DL (elige uno):** `torch` **o** `tensorflow`.  
- Baseline clásico (opcional): `scikit-learn>=1.4,<1.6`, `xgboost`.

### `requirements.txt` de referencia

---
## Instalación en **Windows** (con .BAT incluido)

En Windows, usa el script ya incluido `instalar_dependencias.bat` en la raíz del proyecto. Este script actualiza `pip` e instala todas las dependencias necesarias (incluyendo `streamlit`, `plotly`, `scikit-learn`, `xgboost`, `lightkurve`, `jupyter`, etc.).

### Opción rápida

```powershell
# Desde PowerShell en la carpeta del proyecto
.\instalar_dependencias.bat
```
### (Opcional) Usar entorno virtual `.venv`

Si prefieres aislar dependencias:
```powershell
# Desde PowerShell en la carpeta del proyecto
python -m venv .venv
.\.venv\Scripts\activate
.\instalar_dependencias.bat
```
---
### Artefactos del modelo requeridos

La aplicación busca estos archivos en `./artifacts/` (ver constantes en `app.py`):

- `artifacts/model_xgb.pkl` — Modelo entrenado (por ejemplo, XGBoost).
- `artifacts/scaler.joblib` — Escalador ajustado (por ejemplo, `RobustScaler`).
- `artifacts/feature_columns.json` — Lista de columnas usadas durante el entrenamiento.

Si faltan, la app mostrará un mensaje de error y se detendrá (`st.stop()`), según la lógica de `load_artifacts()` en `app.py`.

---
### Formato de entrada (CSV)

- Sube un **CSV** desde la barra lateral.
- El sistema detecta el delimitador automáticamente y omite líneas corruptas (`on_bad_lines="skip"`).
- Debe incluir las columnas numéricas que espera el modelo/escalador (faltantes se rellenan con `0.0` y se imputan medianas cuando aplique).

---
## Autores

- [Yareth San Miguel Jiménez](https://github.com/yjimenez1) — Ingeniería en Computación
- [Karen Andrea Rodríguez Guevara](https://github.com/kareenarg) — Maestra en Administración de Tecnologías de la Información
- [Jimena Yireh Cano Pérez](https://github.com/yirehcano) — Estudiante
- [Aquiles Piña Olvera](https://github.com/Aquilespina) — Licenciado en Ciencias de la Informática
