import torch
import mvp

import os
import h5py
import numpy as np
from PIL import Image
import datetime
import torchvision.transforms as T
import torchvision.models as models

model = mvp.
model.freeze()
model.to("cuda")

transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

image_path = '/home/venky/CogVLM/test_set/d1.jpg'
image = Image.open(image_path).convert('RGB')
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
preprocessed_image = preprocessed_image.to("cuda")

with torch.no_grad():
    embedding = model(preprocessed_image)
print(embedding.shape)

img_name = os.path.splitext(os.path.basename(image_path))[0]
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"mvp_{img_name}_{current_time}"
run_dir_path = os.path.join('/home/venky/CogVLM/baselines/logs', run_name)
os.makedirs(run_dir_path, exist_ok=True)
with h5py.File(os.path.join(run_dir_path, 'embeddings.h5'), 'w') as f:
    f.create_dataset('embeddings', data=embedding.cpu().numpy())