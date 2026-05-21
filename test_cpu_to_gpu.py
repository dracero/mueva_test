import torch
from colpali_engine.models import ColPali
import traceback

def test():
    print("Loading ColPali on CPU...")
    try:
        model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            device_map=None,  # Load entirely on CPU
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("Success loading on CPU. Moving to GPU...")
        model.to("cuda")
        print("Success moving to GPU!")
        
        # Test an embedding generation!
        print("Testing a forward pass...")
        with torch.no_grad():
            from PIL import Image
            from colpali_engine.models import ColPaliProcessor
            processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            img = Image.new('RGB', (224, 224), color = 'red')
            inputs = processor(text="hello", images=img, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            print("Forward pass successful! Outputs keys:", outputs.keys() if hasattr(outputs, 'keys') else type(outputs))
            
    except Exception as e:
        traceback.print_exc()

test()
