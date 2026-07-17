import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import sys
import os

# --- 1. Define the EXACT Architecture ---
def build_model():
    base_model = models.mobilenet_v2(weights=None)
    in_features = base_model.classifier[1].in_features
    
    base_model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 2)
    )
    return base_model

# --- 2. Image Preprocessing (MATCHING TRAINING EXACTLY) ---
def process_image_to_fft(image_path):
    # 1. Standardization: Read in Grayscale and Resize to 224x224 using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    img_resized = cv2.resize(img, (224, 224))
    
    # 2 & 3. The FFT Conversion and Centering
    f_transform = np.fft.fft2(img_resized)
    f_shift = np.fft.fftshift(f_transform)
    
    # 4. Logarithmic Scaling
    magnitude_spectrum = np.log(1 + np.abs(f_shift))
    
    # 5. OpenCV Normalization (EXACT match to Week 2/3)
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 6. Channel Duplication (The MobileNetV2 Hack)
    final_img_array = np.stack((magnitude_normalized,) * 3, axis=-1)
    
    # Convert to PIL ONLY for the final PyTorch tensor transforms
    fft_image = Image.fromarray(final_img_array)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(fft_image).unsqueeze(0)

# --- 3. Inference Engine ---
def run_inference(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Dual-Domain Inference on: {device}")
    
    model = build_model().to(device)
    
    # ---> THE DICTIONARY FIX <---
    try:
        original_state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for key, value in original_state_dict.items():
            # Strip the 'base_model.' prefix if it exists
            new_key = key.replace('base_model.', '', 1) if key.startswith('base_model.') else key
            new_state_dict[new_key] = value
            
        # ENFORCE STRICT=TRUE to guarantee we are not running on random weights!
        model.load_state_dict(new_state_dict, strict=True)
        print("[+] Custom weights loaded and verified successfully.")
    except Exception as e:
        print(f"[-] CRITICAL FAILURE loading weights: {e}")
        return
        
    model.eval() 
    
    print(f"[*] Preprocessing {os.path.basename(image_path)} via OpenCV FFT...")
    try:
        tensor_img = process_image_to_fft(image_path).to(device)
    except Exception as e:
        print(f"[-] Image Processing Error: {e}")
        return
    
    print("[*] Running network forward pass...")
    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # ---> THE SOFTMAX FLIP FIX <---
        # Alphabetical DataLoader sorting during Week 4: Class 0 = FAKE, Class 1 = REAL
        prob_fake = probabilities[0].item() * 100
        prob_real = probabilities[1].item() * 100
        
        verdict = "FAKE" if prob_fake > prob_real else "REAL"
        confidence = max(prob_real, prob_fake)
        
    print("\n" + "="*40)
    print("      DEEPFAKE DETECTION VERDICT")
    print("="*40)
    print(f"Result     : {verdict}")
    print(f"Confidence : {confidence:.2f}%")
    print("-" * 40)
    print(f"Prob (Fake) : {prob_fake:.2f}%  <-- [Class 0]")
    print(f"Prob (Real) : {prob_real:.2f}%  <-- [Class 1]")
    print("="*40)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 05_inference.py <path_to_image.jpg>")
    else:
        # Update this to point to where your best_model.pth is located
        model_weights_path = "../deepfake_detector_weights.pth"
        if not os.path.exists(model_weights_path):
            print(f"[-] Error: Could not find weights file at {model_weights_path}")
        else:
            run_inference(sys.argv[1], model_weights_path)