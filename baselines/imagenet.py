import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import h5py
import os
import datetime

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = models.resnet50(pretrained=True)  # Ensure pretrained=True to load the ImageNet trained weights
model.eval()
model.to(device)

transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

image_path = '/home/venky/CogVLM/test_set/d5.jpg'
image = Image.open(image_path).convert('RGB')
preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
preprocessed_image = preprocessed_image.to(device)

with torch.no_grad():
    embedding = model(preprocessed_image * 255.0)

print(embedding.shape)

# Saving the embedding, similar to the original script
img_name = os.path.splitext(os.path.basename(image_path))[0]
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"imagenet_{img_name}_{current_time}"
run_dir_path = os.path.join('/home/venky/CogVLM/baselines/logs', run_name)
os.makedirs(run_dir_path, exist_ok=True)
with h5py.File(os.path.join(run_dir_path, 'embeddings.h5'), 'w') as f:
    f.create_dataset('embeddings', data=embedding.cpu().numpy())
