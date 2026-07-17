import torch

weights = torch.load('../deepfake_detector_weights.pth', map_location='cpu')
print("Keys found in saved model:")
for key in weights.keys():
    if 'classifier' in key or 'fc' in key or 'head' in key or 'dense' in key:
        print(f" - {key}")