import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import io

# Import our custom logic!
from core_engine import load_deepfake_model, process_to_fft, GradCAM

st.set_page_config(page_title="Deepfake Forensics", layout="wide")

@st.cache_resource
def init_system():
    # Detect if Streamlit Cloud has a GPU (it usually doesn't on the free tier), fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the weights directly from the root folder
    model = load_deepfake_model('deepfake_detector_weights.pth', device)
    
    # Initialize the Grad-CAM engine
    cam_engine = GradCAM(model, model.features[-1])
    
    return model, cam_engine, device

st.title("🔍 Explainable Dual-Domain Deepfake Detector")
st.markdown("**C-DAC Summer Term Project** | *Frequency-Aware Transfer Learning approach*")
st.markdown("---")

try:
    model, cam_engine, device = init_system()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Ensure 'deepfake_detector_weights.pth' is in the directory.")
    st.stop()

uploaded_file = st.file_uploader("Upload an Image to Analyze", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the raw web bytes directly
    image_bytes = uploaded_file.read()
    
    # Still use PIL just to display the spatial image on the UI
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    spatial_img_resized = pil_image.resize((224, 224))
    
    with st.spinner('Running Dual-Domain Analysis...'):
        # Pass the RAW BYTES to the engine so OpenCV can decode it directly
        tensor_img, original_fft_img = process_to_fft(image_bytes)
        tensor_img = tensor_img.to(device)
        
        heatmap, class_idx, logits = cam_engine.generate_heatmap(tensor_img)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        confidence = probabilities[class_idx].item() * 100
        # FAKE is Class 0 because 'F' comes before 'R' alphabetically
        verdict = "FAKE (Generative Artefacts Detected)" if class_idx == 0 else "REAL (Natural Frequencies)"
        verdict_color = "red" if class_idx == 0 else "green"
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        original_fft_array = np.array(original_fft_img)
        superimposed_fft = heatmap_colored * 0.4 + original_fft_array * 0.6
        superimposed_fft = np.clip(superimposed_fft, 0, 255).astype(np.uint8)

    st.markdown(f"### Verdict: <span style='color:{verdict_color}'>{verdict}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Score:** {confidence:.2f}%")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Spatial Domain")
        st.image(spatial_img_resized, caption="Uploaded Image", width='stretch')
    with col2:
        st.subheader("2. Frequency Spectrum")
        st.image(original_fft_img, caption="FFT Extracted", width='stretch')
    with col3:
        st.subheader("3. Explainability (Grad-CAM)")
        st.image(superimposed_fft, caption="Network Activation Zones", width='stretch')