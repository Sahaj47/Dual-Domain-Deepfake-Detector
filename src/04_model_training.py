import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import time
import copy
import os

# Import the architecture you built in Week 4
from model_architecture import DualDomainDetector

def train_model():
    # -----------------------------------------
    # 1. Configuration & CPU Optimization
    # -----------------------------------------
    # Enforcing CPU as per your hardware constraints
    device = torch.device("cpu")
    print(f"Training initiated on: {device}")

    # Hyperparameters tailored for the Target 90 Run
    BATCH_SIZE = 32  
    EPOCHS = 25             # Increased to allow the scheduler room to operate
    LEARNING_RATE = 0.001   # Back to 0.001 (The scheduler will lower it automatically)
    PATIENCE = 6            # Increased so early stopping doesn't interrupt the scheduler

    # -----------------------------------------
    # 2. DataLoaders & Transforms
    # -----------------------------------------
    # MobileNetV2 relies on ImageNet normalization statistics
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Point these to where your FFT images are saved

    train_dir = os.path.join("../", "data_fft", "train")
    test_dir = os.path.join("../", "data_fft", "test")

    print("Loading datasets into memory...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------
    # 3. Model, Loss, and Optimizer Setup
    # -----------------------------------------
    model = DualDomainDetector(num_classes=2).to(device)
    
    # CrossEntropyLoss safely handles the Softmax math internally
    criterion = nn.CrossEntropyLoss()
    
    # NEW: Added weight_decay to penalize overfitting in the deeper network
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE, weight_decay=1e-4)

    # NEW: The Dynamic Scheduler. Slashes LR by 50% if Val Loss doesn't improve for 2 epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=2)

    # -----------------------------------------
    # 4. The Training Engine & Early Stopping
    # -----------------------------------------
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 20)

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Clear old gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward() # Backpropagation (Calculate new gradients)
            optimizer.step() # Update weights

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        # torch.no_grad() disables gradient calculation to save massive amounts of CPU memory
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(test_dataset)
        
        # Calculate F1-Score based on Week 5 Objectives
        # Calculate F1-Score based on Week 5 Objectives
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        time_elapsed = time.time() - start_time
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1-Score: {epoch_f1:.4f}")
        print(f"Epoch computed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        # NEW: Tell the scheduler to check the Validation Loss
        scheduler.step(epoch_val_loss)

        # --- EARLY STOPPING LOGIC ---
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(">> Validation loss improved. Saving best model weights.")
        else:
            epochs_no_improve += 1
            print(f">> No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= PATIENCE:
                print("\n[!] Early stopping triggered! Halting training to prevent overfitting.")
                break

    # -----------------------------------------
    # 5. Finalize and Save
    # -----------------------------------------
    print("\nTraining complete.")
    model.load_state_dict(best_model_wts)
    
    # Save the finalized "brain" of your project
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/deepfake_detector_weights.pth')
    print("Best model weights successfully saved to 'models/deepfake_detector_weights.pth'")

if __name__ == "__main__":
    train_model()