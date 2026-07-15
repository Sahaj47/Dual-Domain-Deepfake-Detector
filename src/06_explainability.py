import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

# --- 1. Architecture Setup ---
def build_model():
    base_model = models.mobilenet_v2(weights=None) # We don't need ImageNet weights here, we are loading our own
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

# --- 2. Custom Grad-CAM Engine ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval() 
        self.model.zero_grad() 
        
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        score = logits[0][class_idx]
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.sum(self.activations, dim=1).squeeze()
        heatmap = nn.functional.relu(heatmap) 
        
        # SAFETY FIX: Prevent division by zero
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.cpu().numpy(), class_idx, logits

# --- 3. Image Preprocessing (MATCHING TRAINING EXACTLY) ---
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
    
    # 5. OpenCV Normalization (EXACT match to Week 2)
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 6. Channel Duplication (The MobileNetV2 Hack)
    final_img_array = np.stack((magnitude_normalized,) * 3, axis=-1)
    
    # Convert to PIL for PyTorch Transforms
    fft_image = Image.fromarray(final_img_array)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(fft_image).unsqueeze(0), fft_image

# --- 4. Visualizer ---
def visualize_gradcam(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    
    # ---> THE DICTIONARY FIX <---
    original_state_dict = torch.load(model_weights_path, map_location=device)
    new_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key.replace('base_model.', '', 1) if key.startswith('base_model.') else key
        new_state_dict[new_key] = value
        
    # ENFORCE STRICT LOADING
    model.load_state_dict(new_state_dict, strict=True)
    
    target_layer = model.features[-1] 
    
    cam_engine = GradCAM(model, target_layer)
    tensor_img, original_fft_img = process_image_to_fft(image_path)
    tensor_img = tensor_img.to(device)
    
    heatmap, class_idx, logits = cam_engine.generate_heatmap(tensor_img)
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    
    # ---> THE SOFTMAX FLIP FIX <---
    # Remember: Alphabetical sorting means Class 0 = FAKE, Class 1 = REAL
    verdict = "FAKE" if class_idx == 0 else "REAL"
    confidence = probabilities[class_idx].item() * 100
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    original_fft_array = np.array(original_fft_img)
    superimposed_fft = heatmap_colored * 0.4 + original_fft_array * 0.6
    superimposed_fft = np.clip(superimposed_fft, 0, 255).astype(np.uint8)

    spatial_img = Image.open(image_path).convert('RGB').resize((224, 224))
    spatial_array = np.array(spatial_img)
    
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Prediction: {verdict} ({confidence:.2f}%)", fontsize=16, fontweight='bold')
    
    plt.subplot(1, 3, 1)
    plt.imshow(spatial_array)
    plt.title("Original Spatial Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(original_fft_array)
    plt.title("Extracted Frequency Map (FFT)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_fft)
    plt.title("Frequency-Domain Explainability")
    plt.axis('off')
    
    save_name = f"gradcam_result_{os.path.basename(image_path)}"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"[+] Saved Explainability Report to: {save_name}")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 06_explainability.py path/to/image.jpg")
    else:
        model_weights_path = "C:/Users/sahaj/Desktop/cdac_deepfake_project/src/models/deepfake_detector_weights.pth"
        visualize_gradcam(sys.argv[1], model_weights_path)