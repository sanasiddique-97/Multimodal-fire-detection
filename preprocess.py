conda create -n fire-detection python=3.8
conda activate fire-detection
pip install torch torchvision opencv-python pandas matplotlib

import cv2
import os

def resize_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img = cv2.imread(f"{input_dir}/{img_name}")
        img = cv2.resize(img, size)
        cv2.imwrite(f"{output_dir}/{img_name}", img)

resize_images("data/fire", "data/processed/fire")
resize_images("data/non_fire", "data/processed/non_fire")

