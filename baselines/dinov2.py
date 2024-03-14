import torch
import h5py
import os
import datetime
from PIL import Image
import torchvision.transforms as T
import numpy as np

dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dinov2_vitl14.eval()
dinov2_vitl14.to(device)

transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

image_path = '/home/venky/CogVLM/test_set5/IMG_1117.jpg'
image = Image.open(image_path).convert('RGB')
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
preprocessed_image = preprocessed_image.to('cuda')
with torch.no_grad():
  embedding = dinov2_vitl14(preprocessed_image * 255.0)
print(embedding.shape)

img_name = os.path.splitext(os.path.basename(image_path))[0]
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"run_{img_name}_{current_time}"
run_dir_path = os.path.join('/home/venky/CogVLM/baselines/logs_dino', run_name)
os.makedirs(run_dir_path, exist_ok=True)
with h5py.File(os.path.join(run_dir_path, 'embeddings.h5'), 'w') as f:
    f.create_dataset('embeddings', data=embedding.cpu().numpy())
