import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from collections import Counter
from huggingface_hub import InferenceClient

# -------------------------
# Constants
# -------------------------
MODEL_DIR = "models"
MODEL_FILES = ["densenet_model.h5", "resnet_model.h5", "vgg_model.h5", "mobilenet_model.h5"]
CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
    'Spider_mites', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'Healthy'
]

# -------------------------
# Hugging Face Client
# -------------------------
HF_TOKEN = "hf_***"  # enter your token here
client = InferenceClient(provider="novita", api_key=HF_TOKEN)

# -------------------------
# Disease Info
# -------------------------
def get_disease_info(disease_name):
    prompt = f"""
    You are TomatoMedic, an AI assistant for tomato plant health.
    Provide concise, structured information about {disease_name} in this format:

    Symptoms:
    - ...

    Treatment:
    - ...

    Prevention:
    - ...
    """
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Error retrieving info: {e}"

# -------------------------
# Chatbot with memory
# -------------------------
def chatbot_response():
    try:
        messages = [{"role": "system", "content": "You are TomatoMedic, a helpful assistant for tomato plant health."}]
        for role, msg in st.session_state.chat_history:
            messages.append({"role": "user" if role == "You" else "assistant", "content": msg})

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Chatbot error: {e}"

# -------------------------
# Session State
# -------------------------
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'is_leaf' not in st.session_state:
    st.session_state.is_leaf = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Load ML Models
# -------------------------
@st.cache_resource
def load_models():
    models = []
    for file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, file)
        try:
            model = load_model(path, compile=False)
            models.append(model)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return models

models = load_models()

# -------------------------
# Image Processing
# -------------------------
def check_image_quality(image):
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 100:
        return False, f"Image is too blurry (sharpness score: {fm:.1f})."
    if image.size[0] < 224 or image.size[1] < 224:
        return False, "Image resolution too low (min 224x224)."
    return True, "OK"

def detect_tomato_leaf(img):
    img = img.resize((224, 224))
    np_img = np.array(img)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    green_ratio = np.sum(mask > 0) / (224 * 224)
    return green_ratio > 0.1, green_ratio * 100

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(image):
    processed = preprocess_image(image)
    predictions = []
    for model in models:
        try:
            pred = model.predict(processed, verbose=0)
            predicted_index = np.argmax(pred)
            if predicted_index < len(CLASS_NAMES):
                predictions.append(CLASS_NAMES[predicted_index])
        except Exception as e:
            st.error(f"Prediction error: {e}")
    if not predictions:
        return "Prediction failed", []
    return Counter(predictions).most_common(1)[0][0], predictions

# -------------------------
# UI Sections
# -------------------------
def upload_section():
    st.header("Upload or Capture Leaf Image")
    tab1, tab2 = st.tabs(["Upload", "Camera"])
    with tab1:
        uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            ok, msg = check_image_quality(img)
            if not ok:
                st.warning(msg)
                return
            st.session_state.current_image = img
            st.session_state.stage = 'leaf_check'
            st.rerun()
    with tab2:
        cam_image = st.camera_input("Take a photo")
        if cam_image:
            img = Image.open(cam_image)
            ok, msg = check_image_quality(img)
            if not ok:
                st.warning(msg)
                return
            st.session_state.current_image = img
            st.session_state.stage = 'leaf_check'
            st.rerun()

def leaf_check_section():
    img = st.session_state.current_image
    st.header("Leaf Verification")
    st.image(img, use_column_width=True)
    ok, msg = check_image_quality(img)
    if not ok:
        st.warning(msg)
        return
    is_leaf, conf = detect_tomato_leaf(img)
    st.session_state.is_leaf = is_leaf
    if is_leaf:
        st.success("Tomato leaf detected")
        st.session_state.stage = 'analysis'
        st.rerun()
    else:
        st.error("Not a tomato leaf. Upload another image.")

def analysis_section():
    st.header("Disease Analysis")
    img = st.session_state.current_image
    if not st.session_state.analysis_done:
        diagnosis, _ = predict_disease(img)
        if diagnosis == "Prediction failed":
            st.error("Could not analyze. Try another image.")
            return
        st.session_state.results.append({'image': img, 'diagnosis': diagnosis})
        st.session_state.analysis_done = True

    result = st.session_state.results[-1]
    st.subheader(f"Diagnosis: {result['diagnosis']}")
    if result['diagnosis'] != "Healthy":
        with st.spinner("Fetching treatment info..."):
            info = get_disease_info(result['diagnosis'])
            if info:
                st.markdown(f"<div>{info}</div>", unsafe_allow_html=True)
    else:
        st.success("Leaf appears healthy. No treatment needed.")

    # Chat
    st.markdown("---")
    st.subheader("💬 Ask TomatoMedic")

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 {role}:** {msg}")
        else:
            st.markdown(f"**🤖 {role}:** {msg}")

    user_input = st.chat_input("Ask another question about tomato plant health...")
    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        reply = chatbot_response()
        st.session_state.chat_history.append(("TomatoMedic", reply))
        st.rerun()

# -------------------------
# Reset
# -------------------------
def reset_flow():
    st.session_state.stage = 'upload'
    st.session_state.current_image = None
    st.session_state.is_leaf = False
    st.session_state.analysis_done = False
    st.session_state.results = []
    st.session_state.chat_history = []

# -------------------------
# Main
# -------------------------
st.set_page_config(page_title="TomatoMedic", page_icon="🍅")
st.title("TomatoMedic")

if st.session_state.stage == 'upload':
    upload_section()
elif st.session_state.stage == 'leaf_check':
    leaf_check_section()
elif st.session_state.stage == 'analysis':
    analysis_section()

with st.sidebar:
    st.header("Image Tips for Best Results")
    st.markdown("""
    - Use a **clear, centered** tomato leaf.
    - Ensure **bright, natural lighting**.
    - Avoid **blurry or low-res images**.
    - Capture **mostly the leaf**, not background.
    - Keep surroundings **minimal**.
    """)
    if st.button("Reset Session"):
        reset_flow()
        st.rerun()