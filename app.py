# app.py
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# -------------- CONFIG BÁSICA --------------
st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐", layout="wide")

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model_xgb.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
FEATS_PATH = os.path.join(ART_DIR, "feature_columns.json")

PRIMARY = "#6C63FF"
ACCENT = "#22c55e"
DANGER = "#ef4444"

# -------------- ESTILOS GLOBALES --------------
st.markdown(
    f"""
    <style>
      .big-title {{
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
      }}
      .subtitle {{
        color: #444;
        margin-top: 0px;
        margin-bottom: 1rem;
        line-height: 1.4;
      }}
      .soft-card {{
        background: #fff;
        border: 1px solid #eee;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
      }}
      .prob-bar {{
        height: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, {PRIMARY}, #A09BFF);
      }}
      .muted {{
        color: #6b7280; font-size: 0.93rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- I18N (ES / EN) --------------
LANGS = {
    "es": {
        "title": "🪐 Clasificador de Exoplanetas",
        "subtitle": "Sube un archivo CSV con **mediciones numéricas** (características). El sistema calculará la probabilidad de que cada objeto sea un **planeta**. No necesitas experiencia: te guiamos paso a paso.",
        "upload_header": "1) Cargar archivo CSV",
        "upload_help": "Sube un archivo CSV. Si contiene una columna de etiqueta (por ejemplo `Planet` = 0/1), verás métricas de calidad.",
        "example_header": "¿Qué archivo necesito? (ejemplo descargable)",
        "example_text": "- **CSV** con **columnas numéricas** (las mismas usadas para entrenar el modelo).\n- Columna de etiqueta opcional (`Planet`/`planet`) para métricas.\n- Las columnas de texto se ignoran al predecir.\n- Descarga un **ejemplo** para probar.",
        "download_example": "Descargar CSV de ejemplo",
        "prep_header": "2) Preparación automática de datos",
        "prep_shape": "Características alineadas y escaladas",
        "predict_header": "3) Predicción",
        "threshold": "Umbral de probabilidad para 'Planeta'",
        "top20": "Principales 20 por probabilidad",
        "hist": "Distribución de probabilidades",
        "cards_threshold": "Umbral actual",
        "cards_candidates": "Candidatos (≥ umbral)",
        "cards_total": "Total de filas",
        "eval_header": "4) Evaluación (opcional)",
        "no_label_info": "No se encontró columna de etiqueta (por ej. 'Planet' o 'planet'). Agrega una para ver métricas y matriz de confusión.",
        "acc": "Exactitud (Accuracy)",
        "auc": "ROC AUC",
        "confmat_title": "Matriz de confusión (normalizada)",
        "report": "Reporte de clasificación",
        "feat_header": "5) ¿Qué rasgos influyeron más?",
        "feat_info": "No se pudo calcular la importancia de rasgos para este modelo.",
        "download_header": "6) Descargar resultados",
        "download_btn": "Descargar CSV con predicciones",
        "footer": "Consejo: mueve el umbral en la barra lateral para ver cuántos candidatos cambian.",
        "labels_sidebar": "Posibles nombres de la columna de etiqueta",
        "show_confmat": "Mostrar matriz de confusión (si hay etiqueta)",
        "need_artifacts": "No se encontraron artefactos del modelo en ./artifacts/. Exporta: model_xgb.pkl, scaler.joblib, feature_columns.json",
        "preview": "Vista previa del archivo",
        "start_hint": "⬆️ Sube un CSV para comenzar. También puedes descargar el ejemplo.",
        "no_art_cols": "No se pudo mostrar la importancia de rasgos: ",
        "planet": "Planeta",
        "no_planet": "No-Planeta",
    },
    "en": {
        "title": "🪐 Exoplanet Classifier",
        "subtitle": "Upload a CSV with **numerical measurements** (features). The system will estimate the probability that each object is a **planet**. No expertise required: this app guides you step by step.",
        "upload_header": "1) Upload CSV file",
        "upload_help": "Upload a CSV file. If it contains a label column (e.g., `Planet` = 0/1), you will see quality metrics.",
        "example_header": "What file do I need? (downloadable example)",
        "example_text": "- **CSV** with **numeric columns** (same ones used to train the model).\n- Optional label column (`Planet`/`planet`) to show metrics.\n- Text columns are ignored for prediction.\n- Download a **sample** to try it out.",
        "download_example": "Download sample CSV",
        "prep_header": "2) Automatic data preparation",
        "prep_shape": "Aligned & scaled features",
        "predict_header": "3) Prediction",
        "threshold": "Probability threshold for 'Planet'",
        "top20": "Top 20 by probability",
        "hist": "Probability distribution",
        "cards_threshold": "Current threshold",
        "cards_candidates": "Candidates (≥ threshold)",
        "cards_total": "Total rows",
        "eval_header": "4) Evaluation (optional)",
        "no_label_info": "No label column found (e.g., 'Planet' or 'planet'). Add one to see metrics and a confusion matrix.",
        "acc": "Accuracy",
        "auc": "ROC AUC",
        "confmat_title": "Confusion matrix (row-normalized)",
        "report": "Classification report",
        "feat_header": "5) Which features mattered most?",
        "feat_info": "Could not compute feature importance for this model.",
        "download_header": "6) Download results",
        "download_btn": "Download predictions CSV",
        "footer": "Tip: adjust the threshold in the sidebar to see how the number of candidates changes.",
        "labels_sidebar": "Possible label column names",
        "show_confmat": "Show confusion matrix (if label exists)",
        "need_artifacts": "Model artifacts not found in ./artifacts/. Export: model_xgb.pkl, scaler.joblib, feature_columns.json",
        "preview": "File preview",
        "start_hint": "⬆️ Upload a CSV to start. You can also download the example.",
        "no_art_cols": "Could not show feature importance: ",
        "planet": "Planet",
        "no_planet": "Non-Planet",
    }
}

# -------------- SELECTOR DE IDIOMA --------------
with st.sidebar:
    lang = st.selectbox("Language / Idioma", options=["es", "en"], index=0)
T = LANGS[lang]

# -------------- CABECERA --------------
st.markdown(f'<div class="big-title">{T["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{T["subtitle"]}</div>', unsafe_allow_html=True)

# -------------- CARGA DE ARTEFACTOS --------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATS_PATH)):
        st.error(T["need_artifacts"])
        st.stop()
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    with open(FEATS_PATH, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    return model, scaler, feat_cols

model, scaler, FEATURE_COLS = load_artifacts()

# -------------- BLOQUE EJEMPLO DESCARGABLE --------------
with st.expander(T["example_header"]):
    st.markdown(T["example_text"])
    sample = pd.DataFrame({
        "flux_mean": [0.02, -0.15, 0.08, 0.33, -0.05],
        "std_flux": [1.1, 0.9, 1.4, 0.8, 1.2],
        "duration": [2.3, 1.8, 2.9, 3.1, 2.2],
        "depth": [0.001, 0.004, 0.011, 0.030, 0.0006],
        "Planet": [0, 1, 1, 1, 0],
    })
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    st.download_button(T["download_example"], data=buf.getvalue(),
                       file_name="sample_exoplanets.csv", mime="text/csv")

# -------------- SIDEBAR (CONTROLES) --------------
st.sidebar.header("Controls / Controles")
threshold = st.sidebar.slider(T["threshold"], 0.0, 1.0, 0.5, 0.01)
show_confmat = st.sidebar.checkbox(T["show_confmat"], value=True)
label_candidates = st.sidebar.multiselect(
    T["labels_sidebar"],
    options=["Planet", "planet", "label", "label_meta", "koi_disposition", "koi_pdisposition"],
    default=["Planet", "planet"]
)

# -------------- HELPERS --------------
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        # mantenemos strings en otras columnas para devolverlas mezcladas si el usuario quiere
        try:
            out[c] = pd.to_numeric(out[c], errors="ignore")
        except Exception:
            pass
    return out

def align_and_scale(df_raw: pd.DataFrame, scaler, feature_cols: list) -> pd.DataFrame:
    df = coerce_numeric(df_raw)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=df.index)
    return X_scaled

def find_label_column(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------- PASO 1: CARGA DE CSV --------------
st.subheader(T["upload_header"])
st.caption(T["upload_help"])
uploaded = st.file_uploader("CSV", type=["csv"])
if uploaded is None:
    st.info(T["start_hint"])
    st.stop()

df_in = pd.read_csv(uploaded, comment="#")
st.markdown(f"**{T['preview']}**")
st.dataframe(df_in.head(), use_container_width=True)

# -------------- PASO 2: PREPARACIÓN --------------
st.subheader(T["prep_header"])
X_scaled = align_and_scale(df_in, scaler, FEATURE_COLS)
st.write(f"{T['prep_shape']}: {X_scaled.shape}")
st.dataframe(X_scaled.head(), use_container_width=True)

# -------------- PASO 3: PREDICCIÓN --------------
st.subheader(T["predict_header"])
probs = model.predict_proba(X_scaled)[:, 1]
preds = (probs >= threshold).astype(int)

out = pd.DataFrame({"pred_prob": probs, "pred_label": preds}, index=df_in.index)

left, right = st.columns([1, 1])
with left:
    st.markdown(f"**{T['top20']}**")
    top = out.sort_values("pred_prob", ascending=False).head(20).copy()
    top_show = top[["pred_prob", "pred_label"]].copy()
    top_show["pred_label"] = top_show["pred_label"].map({1: T["planet"], 0: T["no_planet"]})
    st.dataframe(top_show, use_container_width=True)

with right:
    st.markdown(f"**{T['hist']}**")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    sns.histplot(probs, bins=30, kde=False, ax=ax)
    ax.set_xlabel("P(Planet)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram")
    st.pyplot(fig)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(f"**{T['cards_threshold']}**")
    st.markdown(f"<h3>{int(threshold*100)}%</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(f"**{T['cards_candidates']}**")
    st.markdown(f"<h3>{int((probs>=threshold).sum())}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(f"**{T['cards_total']}**")
    st.markdown(f"<h3>{len(probs)}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------- PASO 4: EVALUACIÓN (OPCIONAL) --------------
st.subheader(T["eval_header"])
if show_confmat:
    lbl_col = find_label_column(df_in, label_candidates)
    if lbl_col is None:
        st.info(T["no_label_info"])
    else:
        y_true_raw = df_in[lbl_col].astype(str).str.upper().str.strip()
        map_pos = {"1", "PLANET", "TRUE", "PC", "CANDIDATE", "CONFIRMED"}
        y_true = y_true_raw.apply(lambda v: 1 if v in map_pos or v == "1" else 0).values

        from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
        cm = confusion_matrix(y_true, preds)
        acc = accuracy_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = np.nan

        m1, m2 = st.columns(2)
        with m1:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown(f"**{T['acc']}**")
            st.markdown(f"<h3>{acc:.4f}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown(f"**{T['auc']}**")
            st.markdown(f"<h3>{'N/A' if np.isnan(auc) else f'{auc:.4f}'}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig2, ax2 = plt.subplots(figsize=(5.5, 5.0))
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="rocket_r",
                    xticklabels=[T["no_planet"], T["planet"]],
                    yticklabels=[T["no_planet"], T["planet"]],
                    cbar=True, ax=ax2)
        ax2.set_title(T["confmat_title"])
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        st.pyplot(fig2)

        with st.expander(f"📜 {T['report']}"):
            st.text(classification_report(y_true, preds, target_names=[T["no_planet"], T["planet"]]))

# -------------- PASO 5: IMPORTANCIA DE RASGOS --------------
st.subheader(T["feat_header"])
try:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    else:
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        importances = pd.Series(gain).reindex(FEATURE_COLS).fillna(0.0).sort_values(ascending=False)

    if importances is not None and importances.sum() > 0:
        topk = importances.head(20)
        fig3, ax3 = plt.subplots(figsize=(6.5, 6.0))
        topk.sort_values().plot(kind="barh", ax=ax3)
        ax3.set_title("Top-20 Feature Importance")
        ax3.set_xlabel("Importance")
        st.pyplot(fig3)
    else:
        st.info(T["feat_info"])
except Exception as e:
    st.info(T["no_art_cols"] + str(e))

# -------------- PASO 6: DESCARGA --------------
st.subheader(T["download_header"])
out_to_download = pd.concat([df_in.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
csv_bytes = out_to_download.to_csv(index=False).encode("utf-8")
st.download_button(f"⬇️ {T['download_btn']}", data=csv_bytes,
                   file_name="predictions_exoplanets.csv", mime="text/csv")

st.markdown(f'<div class="muted">{T["footer"]}</div>', unsafe_allow_html=True)
