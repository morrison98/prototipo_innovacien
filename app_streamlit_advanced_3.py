import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model


# Datos fijos del repositorio GitHub con modelos .keras
GITHUB_USER = "morrison98"
GITHUB_REPO = "prototipo_innovacien"  # corregido sin guión bajo extra
GITHUB_PATH = ""  # Cambia si tus modelos están en una subcarpeta del repo


@st.cache_data(ttl=600)
def listar_archivos_keras_github(user, repo, path=""):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}"
    r = requests.get(api_url)
    if r.status_code != 200:
        return []
    items = r.json()
    keras_files = [file for file in items if file['name'].endswith('.keras')]
    return [{"name": i["name"], "download_url": i["download_url"]} for i in keras_files]


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


@st.cache_resource(show_spinner=False)
def load_keras_model_safely(path: str):
    try:
        m = load_model(path)
        return m, None
    except Exception as e1:
        try:
            m = load_model(path, compile=False)
            return m, None
        except Exception as e2:
            return None, f"Error al cargar el modelo.\n\nPrimer intento: {repr(e1)}\n\nSegundo intento (compile=False): {repr(e2)}"


st.set_page_config(page_title="Clasificación Histopatología", page_icon="🧫", layout="centered")
st.title("🧫 Clasificación de Histopatología (IDC vs No IDC)")
st.caption("Sube un parche (PNG/JPG). La app cargará tu modelo .keras y preprocesará la imagen de forma consistente.")

# Lista los modelos .keras disponibles en el repo GitHub fijo
modelos = listar_archivos_keras_github(GITHUB_USER, GITHUB_REPO, GITHUB_PATH)
if not modelos:
    st.error("No se encontraron modelos .keras en el repositorio configurado.")
    st.stop()

nombres_modelos = [m['name'] for m in modelos]
modelo_seleccionado = st.selectbox("Selecciona un modelo .keras disponible", nombres_modelos)

modelo_info = next((m for m in modelos if m['name'] == modelo_seleccionado), None)
modelo_local = None
if modelo_info:
    with st.spinner(f"Descargando modelo {modelo_info['name']}..."):
        modelo_local = descargar_modelo_temporal(modelo_info['download_url'])
else:
    st.warning("No se seleccionó un modelo válido.")

threshold = st.slider("Umbral de decisión (probabilidad clase 1)", 0.0, 1.0, 0.50, 0.01)

model, load_err = None, None
if modelo_local:
    model, load_err = load_keras_model_safely(modelo_local)

if load_err:
    st.warning("No se pudo cargar el modelo. Revisa el archivo .keras.")
    with st.expander("Detalles del error"):
        st.code(load_err)
else:
    if model is not None:
        in_shape = model.input_shape
        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 0 and isinstance(in_shape[0], (list, tuple)):
            in_shape = in_shape[0]
        target_h, target_w, target_c = 64, 64, 3
        if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
            target_h = in_shape[1] or target_h
            target_w = in_shape[2] or target_w
            target_c = in_shape[3] or target_c

        st.success("Modelo cargado correctamente.")
        with st.expander("Detalles del modelo"):
            st.write({
                "input_shape_detectado": in_shape,
                "H": target_h,
                "W": target_w,
                "C": target_c
            })
    else:
        st.info("No se pudo cargar el modelo seleccionado.")


def preprocess_image(pil_img: Image.Image, target_hw_c=(64, 64, 3)) -> np.ndarray:
    H, W, C = target_hw_c
    if C == 1:
        img = ImageOps.exif_transpose(pil_img).convert("L").resize((W, H))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
    else:
        img = ImageOps.exif_transpose(pil_img).convert("RGB").resize((W, H))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def prob_class_one(pred: np.ndarray) -> float:
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[0] == 1:
        if pred.shape[1] == 1:
            return float(pred[0, 0])
        if pred.shape[1] == 2:
            return float(pred[0, 1])
    return float(np.ravel(pred)[-1])


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
                "modelo": modelo_seleccionado
            })
    else:
        st.info("Carga un modelo válido para ejecutar la predicción.")

st.markdown("---")
st.markdown("**Consejos:**")
st.markdown(
    "- Si tu modelo fue entrenado en **grises**, la app lo detecta (C=1) y convierte la imagen a L.\n"
    "- Si el tamaño del modelo es distinto a 64×64, la app intenta **leer `input_shape`** para redimensionar correctamente.\n"
    "- El archivo recomendado es **`.keras`** (Keras 3 / TF 2.19+)."
)





