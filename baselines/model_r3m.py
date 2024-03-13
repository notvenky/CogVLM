import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import h5py
import os
import datetime
from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

image_path = '/home/venky/CogVLM/test_set/d5.jpg'
image = Image.open(image_path).convert('RGB')
# image = Image.open(image_path)
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
# preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
with torch.no_grad():
  embedding = r3m(preprocessed_image * 255.0)
print(embedding.shape)

img_name = os.path.splitext(os.path.basename(image_path))[0]
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"run_{img_name}_{current_time}"
run_dir_path = os.path.join('/home/venky/CogVLM/baselines/logs', run_name)
os.makedirs(run_dir_path, exist_ok=True)
with h5py.File(os.path.join(run_dir_path, 'embeddings.h5'), 'w') as f:
    f.create_dataset('embeddings', data=embedding.cpu().numpy())