""" 
Segment Anything Model
In this script, the representations of the image before predicting the mask is extracted 

"""

from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="../../samckpts/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image('/home/venky/CogVLM/test_set5/IMG_1115.jpg')
masks, _, _ = predictor.predict('ball')