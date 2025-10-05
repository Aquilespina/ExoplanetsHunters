import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import base64
from joblib import load
import plotly.graph_objects as go # Usaremos Plotly para gráficos más pro

# -------------- CONFIG BÁSICA DE LA PÁGINA --------------
st.set_page_config(
    page_title="Exoplanets Hunters",
    page_icon="🪐",
    layout="wide" # Layout ancho para el dashboard
)

# ---------------FUNCIÓN PARA CARGAR HOJA DE ESTILOS --------
def cargar_css(archivo_css):
    with open(archivo_css) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

cargar_css("resources/css/estilos_app.css")

# -------------- RUTAS Y CONSTANTES --------------
ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model_xgb.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
FEATS_PATH = os.path.join(ART_DIR, "feature_columns.json")
NASA_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg"
#EXOPLANET_HUNTER_LOGO = "./resources/img/exoplanet-hunters.jpg"
EXOPLANET_IMG_URL = "https://exoplanets.nasa.gov/internal_resources/1633" # Imagen genérica de exoplaneta

# -------------- ESTILOS PERSONALIZADOS (CSS) --------------
st.markdown("""
<style>
    /* Ocultar el menú de Streamlit y el footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Título principal */
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .title-container img {
        width: 80px;
        margin-right: 20px;
    }
    .title-container h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        color: #e2e8f0; /* Color de texto claro */
    }
    
    /* Estilo para las tarjetas de información */
    .detail-card {
        background-color: #1e293b; /* Color de fondo secundario */
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
    }
    
    /* Estilo para las barras de progreso personalizadas */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #0d6efd, #8da0cb);
    }
</style>
""", unsafe_allow_html=True)

# -------------- CARGA DE ARTEFACTOS (CACHED) --------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATS_PATH)):
        st.error("No se encontraron artefactos del modelo. Asegúrate de exportar model_xgb.pkl, scaler.joblib y feature_columns.json en ./artifacts/")
        st.stop()
    model = load(MODEL_PATH)            # XGBClassifier entrenado
    scaler = load(SCALER_PATH)          # RobustScaler ajustado
    with open(FEATS_PATH, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)        # columnas guardadas (fallback)
    return model, scaler, feat_cols
# --- Cargar artefactos (SOLO UNA VEZ) ---
model, scaler, FEATURE_COLS_JSON = load_artifacts()

def get_model_features(model, fallback_cols):
    # 1) sklearn >= 1.0
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        return list(model.feature_names_in_)
    # 2) booster de xgboost
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    # 3) fallback al json
    return list(fallback_cols)

def get_scaler_features(scaler, fallback_cols):
    if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
        return list(scaler.feature_names_in_)
    return list(fallback_cols)

MODEL_FEATURES  = get_model_features(model, FEATURE_COLS_JSON)    # columnas que espera el MODELO (orden)
SCALER_FEATURES = get_scaler_features(scaler, FEATURE_COLS_JSON)  # columnas que espera el SCALER (orden)



# -------------- HELPERS (Funciones de ayuda) --------------
def align_and_scale(df_raw: pd.DataFrame, scaler, feature_cols: list) -> pd.DataFrame:
    df = df_raw.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0 # Rellenar con 0 si falta alguna columna esperada
            
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median()) # Rellenar NaNs con la mediana
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=df.index)
    return X_scaled

# --- Helper crítico: preparar datos para SCALER y para el MODELO ---
def align_and_scale_for_model(df_raw, scaler, scaler_feats, model_feats):
    """
    1) Prepara EXACTAMENTE las columnas que espera el SCALER (scaler_feats):
        - Crea faltantes con 0.0
        - Ordena como scaler_feats
        - Convierte a numérico y rellena NaN con la mediana
        - Aplica scaler.transform(...)
    2) A partir del DataFrame escalado, construye EXACTAMENTE lo que espera el MODELO (model_feats):
        - Toma columnas comunes (mismo orden del modelo)
        - Crea faltantes con 0.0
        - Devuelve DataFrame escalado y matriz NumPy lista para XGBoost (orden model_feats)
    """
    df = df_raw.copy()

    # --- Para el SCALER ---
    for c in scaler_feats:
        if c not in df.columns:
            df[c] = 0.0

    # Solo columnas del scaler y en ese orden
    df_s = df[scaler_feats].apply(pd.to_numeric, errors="coerce")
    # Imputar medianas (solo numéricas)
    df_s = df_s.fillna(df_s.median(numeric_only=True))

    # Validar dimensión esperada por el scaler
    if df_s.shape[1] != len(scaler_feats):
        raise ValueError(
            f"X tiene {df_s.shape[1]} columnas, pero el scaler espera {len(scaler_feats)}. "
            f"Revisa que SCALER_FEATURES esté bien definido."
        )

    # Escalar
    X_scaled = scaler.transform(df_s)
    df_scaled = pd.DataFrame(X_scaled, columns=scaler_feats, index=df.index)

    # --- Para el MODELO ---
    # Columnas comunes ya escaladas
    common = [c for c in model_feats if c in df_scaled.columns]
    X_model_df = df_scaled[common].copy()

    # Columnas que el modelo espera pero no existen en el escalado (añadir 0.0)
    missing_for_model = [c for c in model_feats if c not in df_scaled.columns]
    for c in missing_for_model:
        X_model_df[c] = 0.0

    # Orden EXACTO del modelo
    X_model_df = X_model_df[model_feats]

    # NumPy para evitar validación por nombres en xgboost
    X_model_np = X_model_df.to_numpy(dtype=np.float32, copy=False)

    return df_scaled, X_model_np

# ---------- INFERENCIAS FÍSICAS (moved up) ----------
RSUN_TO_REARTH = 109.2  # 1 R☉ = 109.2 R⊕
ALBEDO_DEFAULT = 0.3

COLS = {
    "teff": ["st_teff", "koi_steff", "teff", "TSTAR"],
    "rstar_rsun": ["st_rad", "koi_srad", "rstar", "RSTAR"],
    "sma_au": ["koi_sma", "sma", "a", "semi_major_axis", "SMA_AU"],
    "insol_searth": ["koi_insol", "insol", "insolation", "S_inc", "flux"],
    "rp_re": ["koi_prad", "planet_radius_earth", "rp_re", "Rp (Re)"],
    "rp_rj": ["planet_radius_jupiter", "rp_rj", "Rp (Rj)"],
    "depth": ["koi_depth", "transit_depth", "depth", "depth_ppm", "DEPTH_PPM"],
}

def _get_first(series_or_dict, keys):
    for k in keys:
        if k in series_or_dict and pd.notnull(series_or_dict[k]):
            val = series_or_dict[k]
            if isinstance(val, (int, float, np.floating)) or (isinstance(val, str) and val.strip() != ""):
                return val
    return None

def infer_insolation_Searth(row):
    """Flujo estelar relativo (S⊕). Usa columna directa si existe; si no,
    aproxima con (Teff/5772)^4 * (R*/a)^2, con R* en R☉ y a en AU."""
    s = _get_first(row, COLS["insol_searth"])
    if s is not None:
        try:
            return float(s)
        except Exception:
            pass

    teff = _get_first(row, COLS["teff"])
    rstar = _get_first(row, COLS["rstar_rsun"])
    a = _get_first(row, COLS["sma_au"])
    try:
        teff = float(teff) if teff is not None else None
        rstar = float(rstar) if rstar is not None else None
        a = float(a) if a is not None else None
    except Exception:
        return None

    if None in (teff, rstar, a) or a <= 0:
        return None
    return (teff / 5772.0) ** 4 * (rstar ** 2) / (a ** 2)

def infer_teq_K(row, albedo=ALBEDO_DEFAULT):
    """Temperatura de equilibrio (K).
    Si hay S⊕: Teq ≈ 278.5 * [(1-A)*S]^(1/4).
    Si no, usa Teq ≈ Teff * sqrt(R*/(2a)) * (1-A)^(1/4)."""
    S = infer_insolation_Searth(row)
    if S is not None and S > 0:
        return 278.5 * ((1 - albedo) * S) ** 0.25

    teff = _get_first(row, COLS["teff"])
    rstar = _get_first(row, COLS["rstar_rsun"])
    a = _get_first(row, COLS["sma_au"])
    try:
        teff = float(teff) if teff is not None else None
        rstar = float(rstar) if rstar is not None else None
        a = float(a) if a is not None else None
    except Exception:
        return None

    if None in (teff, rstar, a) or a <= 0:
        return None
    return teff * np.sqrt(rstar / (2.0 * a)) * ((1 - albedo) ** 0.25)

def infer_radius_Re(row):
    """Radio planetario en radios terrestres.
    Usa columna directa si existe; si no, deriva de profundidad de tránsito y R*."""
    rp_re = _get_first(row, COLS["rp_re"])
    if rp_re is not None:
        try:
            return float(rp_re)
        except Exception:
            pass

    rp_rj = _get_first(row, COLS["rp_rj"])
    if rp_rj is not None:
        try:
            return float(rp_rj) * 11.21  # 1 Rj = 11.21 Re
        except Exception:
            pass

    depth = _get_first(row, COLS["depth"])
    rstar = _get_first(row, COLS["rstar_rsun"])
    try:
        depth = float(depth) if depth is not None else None
        rstar = float(rstar) if rstar is not None else None
    except Exception:
        return None

    if rstar is None or depth is None or depth <= 0:
        return None

    # profundidad puede venir en ppm o fracción:
    if depth > 1:  # asume ppm
        delta = depth / 1e6
    else:
        delta = depth  # ya es fracción

    if delta <= 0:
        return None

    rp_rsun = np.sqrt(delta) * rstar
    return rp_rsun * RSUN_TO_REARTH

def classify_composition(rp_re, teq_k):
    """Clasificación composicional simple + 'probabilidades' heurísticas."""
    if rp_re is None:
        return "Desconocida", [0.0, 0.0, 0.0]  # [rocosa, volátiles, gigante]

    if rp_re < 1.6:
        base = np.array([0.8, 0.2, 0.0])
    elif rp_re < 3.0:
        base = np.array([0.3, 0.6, 0.1])
    else:
        base = np.array([0.05, 0.25, 0.70])

    if teq_k is not None:
        if teq_k < 220 and rp_re < 2.5:
            base += np.array([0.0, 0.10, 0.0])
        if teq_k > 1200 and rp_re < 2.5:
            base += np.array([0.10, -0.10, 0.0])
        if rp_re > 6:
            base = np.array([0.02, 0.08, 0.90])

    base = np.clip(base, 0, None)
    if base.sum() == 0:
        probs = [0.0, 0.0, 0.0]
    else:
        probs = (base / base.sum()).tolist()

    label = (
        "Rocosa" if rp_re < 1.6 else
        "Sub-Neptuno / H2O-rico" if rp_re < 3.0 else
        "Gigante gaseoso"
    )
    return label, probs

def habitable_zone_flag(S):
    if S is None:
        return "Desconocida"
    if 0.35 <= S <= 1.75:
        return "≈ Zona habitable (aprox.)"
    if S < 0.35:
        return "Exterior a HZ"
    return "Interior a HZ"

def likely_minerals(label, teq_k):
    """Listado orientativo de especies/minerales de nubes/superficie."""
    if teq_k is None:
        teq_k = 0
    if label == "Gigante gaseoso":
        if teq_k < 200:
            return "Hielo de CH₄/NH₃; nubes de NH₃; H/He dominante"
        if teq_k < 800:
            return "Nubes de H₂O/NH₄SH; H/He dominante"
        if teq_k < 1400:
            return "Na₂S/KCl posibles; H/He dominante"
        return "Nubes silicatadas (MgSiO₃), Fe; H/He dominante"
    elif "Rocosa" in label:
        if teq_k < 250:
            return "Hielo de H₂O/CO₂; silicatos y Fe"
        if teq_k < 700:
            return "Silicatos (olivino, piroxeno), Fe; posible H₂O"
        if teq_k < 1400:
            return "Silicatos deshidratados; óxidos (FeO, Al₂O₃)"
        return "Superficies ultracalientes: evaporitas, SiO; posibles vapores metálicos"
    else:  # Sub-Neptuno
        if teq_k < 300:
            return "H₂O/NH₃/CH₄; mezcla roca-hielo"
        if teq_k < 1000:
            return "H₂O/volátiles; brumas orgánicas"
        return "Atmósferas tenues; posibles nubes de sales (KCl/Na₂S)"

# Inicializar estado de sesión para guardar el planeta seleccionado
if 'selected_planet_idx' not in st.session_state:
    st.session_state.selected_planet_idx = None

# -------------- HEADER DE LA APLICACIÓN --------------
ruta_img = os.path.join(os.path.dirname(__file__), "resources", "img", "exoplanet-hunters.jpg")
with open(ruta_img, "rb") as f:
    datos = f.read()
exoplanet_hunters_ruta_img = base64.b64encode(datos).decode()

ruta_img_planets = os.path.join(os.path.dirname(__file__), "resources", "img", "planets.png")
with open(ruta_img_planets, "rb") as f:
    datos = f.read()
planets_img = base64.b64encode(datos).decode()

st.markdown(f"""
<div class="title-container">
    <img src="{NASA_LOGO_URL}" alt="NASA Logo">
    <img src="data:image/png;base64,{exoplanet_hunters_ruta_img}" width="150">
    <h1>Exoplanets Hunters</h1>
    <div class="navbar">
            <div class="nav-left">
                <img src="data:image/png;base64,{planets_img}" width="50">
                <a href="#">HOME</a>
                <a href="#">BROWSE DESTINATIONS</a>
                <a href="#">MISSIONS</a>
            </div>
            <div class="nav-right">
                <span class="search-icon">🔍</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# -------------- ZONA DE CARGA DE ARCHIVO (SIDEBAR) --------------
st.sidebar.header("🚀 Fase 1: Cargar Datos")

uploaded = st.sidebar.file_uploader(
    "📂 Sube tu archivo CSV con datos de candidatos a exoplanetas.",
    type=["csv"],
    help="El archivo debe contener las columnas numéricas utilizadas para el modelo. "
         "Las filas corruptas o con errores serán ignoradas automáticamente."
)

if uploaded is None:
    st.info("🛰️ **Bienvenido a Exoplanets Hunters.** Carga un archivo CSV en la barra lateral para comenzar el análisis.")
    st.stop()

# Intentar leer el CSV de forma segura
try:
    import csv

    # Detectar automáticamente el delimitador
    sample = uploaded.read(4096).decode("utf-8", errors="ignore")
    uploaded.seek(0)
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter if sample else ","

    # Leer con manejo de errores
    df_in = pd.read_csv(
        uploaded,
        sep=delimiter,
        comment="#",
        on_bad_lines="skip",  # ignora filas con columnas inconsistentes
        low_memory=False
    )

    st.success(f"✅ Archivo cargado correctamente con delimitador detectado: '{delimiter}'")
    st.write("**Dimensiones del archivo:**", df_in.shape)
    st.dataframe(df_in.head())

except Exception as e:
    st.error("⚠️ Ocurrió un error al leer el CSV. Revisa el formato del archivo.")
    st.exception(e)
    st.stop()


# --- Realizar predicciones (corazón de la app) ---
# Preprocesar correctamente con las columnas del scaler y del modelo
df_scaled, X_np = align_and_scale_for_model(
    df_in,
    scaler,
    scaler_feats=SCALER_FEATURES,   # columnas usadas por el scaler
    model_feats=MODEL_FEATURES      # columnas esperadas por el modelo
)

# Calcular probabilidades de clase con el modelo entrenado
# Evita errores de nombres con XGBoost usando inplace_predict sobre NumPy:
probs = model.get_booster().inplace_predict(X_np)
preds = (probs >= 0.5).astype(int)  # o ajusta el umbral desde el sidebar si lo agregas

# Crear un DataFrame con los resultados
df_results = df_in.copy()
df_results['probability'] = probs
df_results['pred'] = preds
df_results = df_results.sort_values('probability', ascending=False).reset_index(drop=True)

# -------------------- DASHBOARD PRINCIPAL --------------------
st.markdown("---")
col1, col2 = st.columns([0.6, 0.4]) # Dividir en dos columnas

# --- COLUMNA IZQUIERDA: VISUALIZACIÓN DE DATOS ---
with col1:
    st.subheader("🔭 Explorador de Características")
    
    # Selectores para el gráfico de dispersión
    numeric_cols = df_in.select_dtypes(include=np.number).columns.tolist()
    
    # Asegurarse de que haya al menos 2 columnas numéricas
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Eje X", options=numeric_cols, index=0)
        y_axis = st.selectbox("Eje Y", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        # Crear gráfico de dispersión con Plotly
        fig = go.Figure(data=go.Scatter(
            x=df_results[x_axis],
            y=df_results[y_axis],
            mode='markers',
            marker=dict(
                size=8,
                color=df_results['probability'], # Color basado en la probabilidad
                colorscale='Viridis', # Paleta de colores
                showscale=True,
                colorbar=dict(title="Probabilidad")
            ),
            text=[f"Prob: {p:.2%}" for p in df_results['probability']], # Tooltip
            hoverinfo='text+x+y'
        ))

        fig.update_layout(
            title=f'Relación entre {x_axis} y {y_axis}',
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template='plotly_dark', # Tema oscuro para el gráfico
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("El archivo CSV no contiene suficientes columnas numéricas para generar un gráfico de dispersión.")

    # Detalle del planeta seleccionado (se muestra aquí)
    st.subheader("🌌 Ficha del Candidato Seleccionado")
    if st.session_state.selected_planet_idx is not None:
        planet_data = df_results.loc[st.session_state.selected_planet_idx]
        prob_percent = planet_data['probability'] * 100
        
        with st.container():
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            
            c1, c2 = st.columns([0.4, 0.6])
            with c1:
                exoplanet_img = os.path.join(os.path.dirname(__file__), "resources", "img", "exoplanet.jpg")
                st.image(exoplanet_img, caption=f"Recreación artística")
            
            with c2:
                st.markdown(f"#### Candidato ID: {st.session_state.selected_planet_idx}")
                st.markdown(f"#### Hostname: {st.session_state.selected_planet_idx}")
                st.markdown(f"### Probabilidad de ser Exoplaneta:")
                st.markdown(f"<h1 style='color: #8da0cb;'>{prob_percent:.2f}%</h1>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Datos principales:**")
            
            # Mostrar algunas de las características más importantes
            display_cols = [col for col in planet_data.index if col != 'probability' and col in numeric_cols[:5]]
            st.dataframe(planet_data[display_cols], use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            # === NUEVO: cálculos de inferencia para este candidato ===
            planet_data = df_results.loc[st.session_state.selected_planet_idx]
            prob_percent = planet_data['probability'] * 100

            S_earth = infer_insolation_Searth(planet_data)      # flujo relativo a la Tierra
            Teq = infer_teq_K(planet_data)                      # K
            Rp_Re = infer_radius_Re(planet_data)                # radios terrestres
            comp_label, comp_probs = classify_composition(Rp_Re, Teq)
            hz_flag = habitable_zone_flag(S_earth)
            minerals = likely_minerals(comp_label, Teq)

            # Normalizaciones para evitar 'nan'
            S_text = f"{S_earth:.2f} S⊕" if S_earth is not None else "N/D"
            Teq_text = f"{Teq:.0f} K" if Teq is not None else "N/D"
            Rp_text = f"{Rp_Re:.2f} R⊕" if Rp_Re is not None else "N/D"

            # === GRÁFICO: Gauge de Temperatura de Equilibrio ===
            if Teq is not None and np.isfinite(Teq):
                fig_teq = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(Teq),
                    number={"suffix": " K"},
                    gauge={
                        "axis": {"range": [0, 3000]},
                        "bar": {"thickness": 0.3},
                        "threshold": {"line": {"width": 2}, "thickness": 0.75, "value": float(Teq)}
                    },
                    title={"text": "Temperatura de equilibrio"}
                ))
                fig_teq.update_layout(template="plotly_dark", height=250, margin=dict(t=40, b=20, l=10, r=10))
            else:
                fig_teq = None

            # === GRÁFICO: Barras de prob. composicional ===
            comp_cats = ["Rocosa", "Volátiles", "Gigante"]
            fig_comp = go.Figure(go.Bar(
                x=comp_cats, y=comp_probs,
                text=[f"{p*100:.0f}%" for p in comp_probs],
                textposition="outside"
            ))
            fig_comp.update_layout(
                title="Composición probable (heurística)",
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                template="plotly_dark",
                height=260, margin=dict(t=40, b=20, l=10, r=10)
            )

            # === UI: mostramos tarjetas con los resultados ===
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            cTop1, cTop2 = st.columns([0.5, 0.5])

            with cTop1:
                if fig_teq:
                    st.plotly_chart(fig_teq, use_container_width=True)
                st.markdown("**Flujo estelar relativo:** " + S_text)
                st.markdown("**Radio estimado:** " + Rp_text)
                st.markdown("**Zona habitable (aprox.):** " + hz_flag)

            with cTop2:
                st.plotly_chart(fig_comp, use_container_width=True)
                st.markdown("**Clase:** " + comp_label)
                st.markdown("**Minerales/Especies probables:**")
                st.markdown(f"- {minerals}")

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Selecciona un candidato de la lista de la derecha para ver sus detalles.")

# --- COLUMNA DERECHA: RANKING DE CANDIDATOS ---
with col2:
    st.subheader("🏆 Ranking de Candidatos")

    # (opcional) este slider puede estar fuera del with col2 sin problema
    top_n = st.sidebar.slider("Mostrar Top N candidatos", 5, 50, 15)

    for idx, row in df_results.head(top_n).iterrows():
        prob = float(row['probability'])
        prob_percent = prob * 100

        c1, c2, c3 = st.columns([0.4, 0.4, 0.2])
        with c1:
            st.markdown(f"**ID: {idx}**")
        with c2:
            # Para evitar problemas de rango, usa entero 0-100
            st.progress(int(prob_percent), text=f"{prob_percent:.2f}%")
        with c3:
            if st.button("Ver", key=f"btn_{idx}"):
                st.session_state.selected_planet_idx = idx
                st.rerun()  # ← reemplaza experimental_rerun por rerun


# --- SECCIÓN DE DESCARGA (EN SIDEBAR) ---

# --- SECCIÓN DE DESCARGA (EN SIDEBAR) ---
st.sidebar.header("📁 Fase 2: Exportar")
csv_bytes = df_results.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Descargar Resultados con Predicciones",
    data=csv_bytes,
    file_name="exoplanet_predictions.csv",
    mime="text/csv",
)