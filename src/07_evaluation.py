import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import os
import sys

# --- 1. Standard DataLoader Transform (NO FFT Math) ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. Architecture Setup ---
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

# --- 3. Evaluation Loop ---
def evaluate_model(test_dir, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Fast Evaluation on: {device}")
    
    # Load Model
    model = build_model().to(device)
    
    # ---> THE FIX: INTERCEPT AND RENAME THE KEYS <---
    original_state_dict = torch.load(model_weights_path, map_location=device)
    new_state_dict = {}
    
    for key, value in original_state_dict.items():
        # Strip the 'base_model.' prefix if it exists
        if key.startswith('base_model.'):
            new_key = key.replace('base_model.', '', 1)
        else:
            new_key = key
        new_state_dict[new_key] = value
        
    # Load the mapped weights with strict=True to guarantee a perfect fit!
    model.load_state_dict(new_state_dict, strict=True)
    print("[+] Weights successfully mapped and loaded into the architecture.")
    
    model.eval()
    
    # Load Dataset Directly
    print(f"[*] Loading Pre-processed FFT Dataset from: {test_dir}")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    y_true = []
    y_pred = []
    y_prob = []
    
    print("[*] Processing pre-calculated FFT images...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_prob.extend(probs[:, 0].cpu().numpy()) # Capturing Class 0 (Fake)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            if (i+1) % 10 == 0:
                print(f"    - Processed batch {i+1}/{len(test_loader)}")

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print("         FINAL TEST SET METRICS")
    print("="*40)
    print(f"Total Images : {len(y_true)}")
    print(f"Accuracy     : {acc*100:.2f}%")
    print(f"Precision    : {prec*100:.2f}%")
    print(f"Recall       : {rec*100:.2f}%")
    print(f"F1-Score     : {f1*100:.2f}%")
    print("="*40)

    # Plot Confusion Matrix
    class_names = test_dataset.classes
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Deepfake Detection Confusion Matrix (Test Set)', pad=20)
    plt.ylabel('Actual Truth')
    plt.xlabel('Model Prediction')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png', dpi=300)
    print("\n[+] Saved Confusion Matrix graphic as 'final_confusion_matrix.png'")

    # Calculate AUC Score (Invert y_true so scikit-learn knows 0 is the target)
    y_true_inverted = 1 - np.array(y_true)
    auc_score = roc_auc_score(y_true_inverted, y_prob)
    print(f"AUC Score    : {auc_score*100:.2f}%")
    print("="*40)

    # Generate ROC Curve Graphic
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=0)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'Model ROC curve (area = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('final_roc_curve.png', dpi=300)
    print("[+] Saved ROC Curve graphic as 'final_roc_curve.png'")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python 07_evaluation.py <path_to_data_fft_folder> <path_to_model.pth>")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])