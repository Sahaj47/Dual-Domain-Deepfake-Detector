import os
import cv2
import numpy as np

def process_and_save_fft(input_dir, output_dir):
    """
    Reads spatial images, converts to FFT magnitude spectrums, 
    and saves them as 3-channel images for MobileNetV2.
    """
    # Create the output directory structure if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We only want to process actual image files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    # Track progress
    processed_count = 0
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(input_dir, filename)
        
        # 1. Standardization: Read in Grayscale and Resize to 224x224
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue
            
        img_resized = cv2.resize(img, (224, 224))
        
        # 2 & 3. The FFT Conversion and Centering
        f_transform = np.fft.fft2(img_resized)
        f_shift = np.fft.fftshift(f_transform)
        
        # 4. Logarithmic Scaling
        # We add 1 to avoid log(0) errors
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Normalize the astronomical raw numbers down to 0-255 (standard pixel values)
        magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 5. Channel Duplication (The MobileNetV2 Hack)
        # Stack the 1-channel grayscale image 3 times to mimic RGB
        final_img = np.stack((magnitude_normalized,) * 3, axis=-1)
        
        # 6. Save to Disk
        # We strictly save as .png to prevent JPEG compression from ruining the frequency data we just extracted
        output_filename = os.path.splitext(filename)[0] + "_fft.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, final_img)
        
        processed_count += 1
        
    print(f"  -> Successfully converted {processed_count} images in {input_dir}")

# --- Execution Block ---
if __name__ == "__main__":
    print("Starting Dual-Domain Preprocessing Pipeline...")
    
    # Define your base paths based on the Week 2 structure
    base_input_dir = r"../data/"
    base_output_dir = r"../data_fft/"
    
    splits = ["train", "test"]
    classes = ["REAL", "FAKE"]

    for split in splits:
        for cls in classes:
            in_dir = os.path.join(base_input_dir, split, cls)
            out_dir = os.path.join(base_output_dir, split, cls)
            
            print(f"\nProcessing directory: {split}/{cls}")
            process_and_save_fft(in_dir, out_dir)
            
    print("\nPipeline Complete! All 10,000 images are now in the frequency domain.")