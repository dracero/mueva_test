import torch
import sys
from PIL import Image
from transformers import BitsAndBytesConfig
from colpali_engine.models import ColPali as ColPaliModel
from colpali_engine.models import ColPaliProcessor

try:
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    model = ColPaliModel.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # create dummy 400x300 image
    img1 = Image.new('RGB', (400, 300))
    # create dummy 1000x1000 image
    img2 = Image.new('RGB', (1000, 1000))
    
    b1 = processor.process_images([img1])
    b2 = processor.process_images([img2])
    
    e1 = model(**b1)
    e2 = model(**b2)
    print("Tokens for 400x300:", e1[0].shape)
    print("Tokens for 1000x1000:", e2[0].shape)
except Exception as e:
    print(e)
