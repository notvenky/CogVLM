import os
import torch
from PIL import Image
from lang_sam import LangSAM
import numpy as np
import datetime
import h5py

sweep_folder = "/home/venky/CogVLM/test_sets/test_set5"
text_prompt = "table"

def save_image_representation(image_representation, base_folder, date_str, frame_number):
    log_folder = os.path.join(base_folder, date_str, frame_number)
    os.makedirs(log_folder, exist_ok=True)
    h5_filename = os.path.join(log_folder, f"{frame_number}_representation.h5")
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset("image_representation", data=image_representation)
    return h5_filename

def process_image(image_path, text_prompt):
    image_pil = Image.open(image_path).convert("RGB")
    model = LangSAM()
    masks, boxes, phrases, logits, image_representation = model.predict(image_pil, text_prompt)
    return masks, boxes, phrases, logits, image_representation

def main(sweep_folder, text_prompt):
    base_log_folder = "sam_logs"
    os.makedirs(base_log_folder, exist_ok=True)

    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    image_files = [f for f in os.listdir(sweep_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort files based on frame number extracted from the file name
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for image_file in image_files:
        image_path = os.path.join(sweep_folder, image_file)
        if os.path.isfile(image_path):
            frame_number = ''.join(filter(str.isdigit, os.path.splitext(image_file)[0]))
            print(f"Processing {image_path}...")
            masks, boxes, phrases, logits, image_representation = process_image(image_path, text_prompt)
            log_file = save_image_representation(image_representation, base_log_folder, date_str, frame_number)
            print(f"Saved image representation to {log_file}")

if __name__ == "__main__":
    main(sweep_folder, text_prompt)