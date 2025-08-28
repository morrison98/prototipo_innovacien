# app_streamlit.py
# ------------------------------------------------------------
# App Streamlit para usar un modelo Keras (.keras) y clasificar
# parches de histopatología (IDC vs No IDC).
#
# Ejecutar:
#   streamlit run app_streamlit.py
#
# Dependencias:
#   pip install streamlit tensorflow pillow numpy requests
# ------------------------------------------------------------

import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import requests
import tempfile

import tensorflow as tf
from tensorflow.keras.models import load_model


# ---------------------- Función para listar archivos .keras en GitHub ----------------------
def listar_archivos_keras_github(user, repo, path=""):
    """
    Lista archivos .keras en el repositorio GitHub dado, en la carpeta `path` (raíz por defecto).
    Retorna lista de dicts: {'name': archivo, 'download_url': url_cruda}
    """
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}"
    r = requests.get(api_url)
    if r.status_code != 200:
        return []
    items = r.json()
    keras_files = []
    for i in items:
        if i["type"] == "file" and i["name"].endswith(".keras"):
            keras_files.append({
                "name": i["name"],
                "download_url": i["download_url"]
            })
    return keras_files

# ---------------------- Función para descargar modelo temporalmente ----------------------
def descargar_modelo_temporal(url):
    r = requests.get(url)
    if r.status_code == 200:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    else:
        st.error("Error descargando el modelo.")
        return None

# ---------------------- Configuración UI ----------------------
st.set_page_config(page_title="Clasificación Histopatología", page_icon="🧫", layout="centered")
st.title("🧫 Clasificación de Histopatología (IDC vs No IDC)")
st.caption("Sube un parche (PNG/JPG). La app cargará tu modelo .keras y preprocesará la imagen de forma consistente.")

# ---------------------- Sidebar ----------------------
st.sidebar.header("⚙️ Configuración")

# Parámetros para GitHub
github_user = st.sidebar.text_input("Usuario GitHub", value="")
github_repo = st.sidebar.text_input("Repositorio GitHub", value="")
github_path = st.sidebar.text_input("Path en repo para buscar modelos (.keras)", value="")

if github_user and github_repo:
    if st.sidebar.button("Buscar modelos .keras en GitHub"):
        modelos = listar_archivos_keras_github(github_user, github_repo, github_path)
        if modelos:
            nombres_modelos = [m['name'] for m in modelos]
            modelo_seleccionado = st.sidebar.selectbox("Selecciona un modelo .keras", nombres_modelos)

            modelo_info = next((m for m in modelos if m['name'] == modelo_seleccionado), None)
            if modelo_info:
                st.sidebar.markdown(f"Descargando modelo **{modelo_info['name']}** ...")
                modelo_local = descargar_modelo_temporal(modelo_info['download_url'])
                if modelo_local:
                    model_path = modelo_local
                else:
                    model_path = None
        else:
            st.sidebar.warning("No se encontraron archivos .keras en el repo/path indicado.")
            model_path = st.sidebar.text_input("Ruta al modelo (.keras)", value="modelo_cancer_37_83.keras")
    else:
        # Si no busca, permite ingresar ruta local manualmente
        model_path = st.sidebar.text_input("Ruta al modelo (.keras)", value="modelo_cancer_37_83.keras")
else:
    # Si no llena GitHub, ingreso manual ruta local
    model_path = st.sidebar.text_input("Ruta al modelo (.keras)", value="modelo_cancer_37_83.keras")

threshold = st.sidebar.slider("Umbral de decisión (probabilidad clase 1)", 0.0, 1.0, 0.50, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Notas**")
st.sidebar.markdown(
    "- El modelo debe ser un clasificador binario con salida `sigmoid` (1 neurona) **o** softmax de 2 clases.\n"
    "- Si tu modelo fue entrenado con imágenes en escala de grises, la app lo detecta y convierte a 1 canal."
)

# ---------------------- Carga del modelo ----------------------
@st.cache_resource(show_spinner=False)
def load_keras_model_safely(path: str):
    """
    Intenta cargar un modelo Keras (.keras).
    - Primero intenta con compile=True (restaura optimizador/estado si está).
    - Si falla (p.ej. objetos custom), reintenta con compile=False.
    """
    try:
        m = load_model(path)
        return m, None
    except Exception as e1:
        try:
            m = load_model(path, compile=False)
            return m, None
        except Exception as e2:
            return None, f"Error al cargar el modelo.\n\nPrimer intento: {repr(e1)}\n\nSegundo intento (compile=False): {repr(e2)}"


model, load_err = None, None
if model_path:
    model, load_err = load_keras_model_safely(model_path)

if load_err:
    st.warning("No se pudo cargar el modelo. Revisa la ruta o el archivo .keras.")
    with st.expander("Detalles del error"):
        st.code(load_err)
else:
    if model is not None:
        # Mostrar info del modelo
        try:
            in_shape = model.input_shape  # (None, H, W, C) o lista si tiene múltiples entradas
        except Exception:
            in_shape = None

        # Detectar input principal si es lista
        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 0 and isinstance(in_shape[0], (list, tuple)):
            # tomar la primera entrada
            in_shape = in_shape[0]

        # Valores por defecto si el modelo no expone input_shape claramente
        target_h, target_w, target_c = 64, 64, 3
        if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
            # in_shape: (None, H, W, C)
            target_h = in_shape[1] if in_shape[1] is not None else target_h
            target_w = in_shape[2] if in_shape[2] is not None else target_w
            target_c = in_shape[3] if in_shape[3] is not None else target_c

        st.success("Modelo cargado correctamente.")
        with st.expander("Detalles del modelo"):
            st.write({
                "input_shape_detectado": in_shape,
                "H": target_h,
                "W": target_w,
                "C": target_c
            })
    else:
        st.info("Carga un modelo válido para ejecutar la predicción.")

# ---------------------- Utilidades de preprocesamiento ----------------------
def preprocess_image(pil_img: Image.Image, target_hw_c=(64, 64, 3)) -> np.ndarray:
    """
    Preprocesa la imagen para coincidir con el input del modelo:
    - Convierte a RGB o L (si C=1)
    - Redimensiona a (H, W)
    - Normaliza a [0, 1]
    - Agrega batch dimension: (1, H, W, C)
    """
    H, W, C = target_hw_c

    if C == 1:
        # Escala de grises
        img = ImageOps.exif_transpose(pil_img).convert("L").resize((W, H))
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W)
        arr = np.expand_dims(arr, axis=-1)                # (H, W, 1)
    else:
        # Color
        img = ImageOps.exif_transpose(pil_img).convert("RGB").resize((W, H))
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)

    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr


def prob_class_one(pred: np.ndarray) -> float:
    """
    Devuelve la probabilidad de la clase 1:
    - Si salida es shape (1,1): se asume sigmoid -> p = y[0][0]
    - Si salida es shape (1,2): se asume softmax -> p = y[0][1]
    - Si la forma no es ninguna de las anteriores, intenta colapsar.
    """
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[0] == 1:
        if pred.shape[1] == 1:
            return float(pred[0, 0])  # sigmoid
        if pred.shape[1] == 2:
            return float(pred[0, 1])  # softmax clase 1
    # Fallback (por si hay formas raras)
    return float(np.ravel(pred)[-1])


# ---------------------- Carga de imagen e inferencia ----------------------
uploaded = st.file_uploader("Sube una imagen (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"No se pudo leer la imagen: {e}")
        st.stop()

    st.image(image, caption="Imagen cargada", use_column_width=True)

    if model is not None:
        with st.spinner("Realizando inferencia..."):
            x = preprocess_image(image, (target_h, target_w, target_c))
            y = model.predict(x, verbose=0)
            p1 = prob_class_one(y)
            pred_label = int(p1 >= threshold)

        st.subheader("Resultado")
        st.metric("Probabilidad de clase 1 (IDC)", f"{p1:.4f}")
        st.progress(min(max(p1, 0.0), 1.0))
        st.write(f"**Predicción binaria (umbral={threshold:.2f}):** `{'1 (IDC)' if pred_label==1 else '0 (No IDC)'}`")

        with st.expander("Detalles técnicos"):
            st.write({
                "input_shape_usado": (1, target_h, target_w, target_c),
                "salida_modelo_shape": np.asarray(y).shape,
                "modelo": model_path
            })
    else:
        st.info("Carga un modelo válido para ejecutar la predicción.")

# ---------------------- Ayuda ----------------------
st.markdown("---")
st.markdown("**Consejos:**")
st.markdown(
    "- Si tu modelo fue entrenado en **grises**, la app lo detecta (C=1) y convierte la imagen a L.\n"
    "- Si el tamaño del modelo es distinto a 64×64, la app intenta **leer `input_shape`** para redimensionar correctamente.\n"
    "- El archivo recomendado es **`.keras`** (Keras 3 / TF 2.19+)."
)
