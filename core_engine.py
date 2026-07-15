import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

# --- 1. Architecture ---
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

# --- 2. Safe Weight Loader ---
def load_deepfake_model(weights_path, device):
    model = build_model().to(device)
    original_state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = {}
    
    # Strip the prefix to ensure the brain actually loads
    for key, value in original_state_dict.items():
        new_key = key.replace('base_model.', '', 1) if key.startswith('base_model.') else key
        new_state_dict[new_key] = value
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

# --- 3. FFT Preprocessing ---
def process_to_fft(image_bytes):
    # 1. Decode raw web bytes EXACTLY like cv2.imread(..., cv2.IMREAD_GRAYSCALE)
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # 2. Resize using OpenCV (Bilinear interpolation)
    img_resized = cv2.resize(img_gray, (224, 224))
    
    # 3. FFT Conversion and Centering
    f_transform = np.fft.fft2(img_resized)
    f_shift = np.fft.fftshift(f_transform)
    
    # 4. Logarithmic Scaling
    magnitude_spectrum = np.log(1 + np.abs(f_shift))
    
    # 5. OpenCV Normalization (EXACT match to Week 2)
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 6. Channel Duplication (The MobileNetV2 Hack)
    final_img_array = np.stack((magnitude_normalized,) * 3, axis=-1)
    
    # 7. Convert back to PIL only for the final standard PyTorch transforms
    fft_image = Image.fromarray(final_img_array)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(fft_image).unsqueeze(0), fft_image

# --- 4. Grad-CAM Class ---
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
        
        # SAFETY FIX: Prevent division by zero if all gradients are negative
        heatmap /= (torch.max(heatmap) + 1e-8) 
        
        return heatmap.cpu().numpy(), class_idx, logits