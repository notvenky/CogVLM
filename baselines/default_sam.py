import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_h"](checkpoint="../../samckpts/sam_vit_h_4b8939.pth")
sam = sam.to(device)
predictor = SamPredictor(sam)
img_path = '../test_set5/IMG_1115.jpg'
# image format should be in RGB
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
masks, _, _ = predictor.predict('find the ball')