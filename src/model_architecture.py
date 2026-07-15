import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class DualDomainDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(DualDomainDetector, self).__init__()
        
        # 1. Load the pre-trained MobileNetV2 with ImageNet weights
        print("Loading pre-trained MobileNetV2...")
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # 2. Freeze all base convolutional layers
        # This stops the CPU from calculating gradients for millions of parameters
        for param in self.base_model.features.parameters():
            param.requires_grad = False
            
        # 3. Attach a lightweight classification head
        # MobileNetV2's final feature map outputs 1280 channels
        # 3. Attach a HIGH-CAPACITY classification head
        in_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),               # Slightly lower initial dropout
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),             # NEW: Stabilizes mathematical variance
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),             # NEW: Second hidden layer for complex patterns
            nn.BatchNorm1d(128),             # NEW: Stabilizes the second layer
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Pass the 3-channel frequency images through the network
        return self.base_model(x)

# --- Quick Validation Test ---
if __name__ == "__main__":
    # Initialize the model
    model = DualDomainDetector(num_classes=2)
    print("Model initialized successfully!")
    
    # Create a dummy tensor representing a single batch of one 224x224 RGB image
    # Shape: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Pass the dummy tensor through the model to ensure it doesn't crash
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape} -> (Batch Size, Num Classes)")
    print("Architecture is CPU-ready and fully functional.")