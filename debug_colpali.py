
import torch
from colpali_engine.models import ColPali as ColPaliModel
from colpali_engine.models import ColPaliProcessor
import traceback

def test_colpali_loading():
    print("Testing ColPali loading...")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        
        print("Loading model with 4-bit quantization...")
        model = ColPaliModel.from_pretrained(
            "vidore/colpali-v1.2",
            quantization_config=quantization_config,
            device_map=device
        )
        print("Model loaded.")
        
        print("Loading processor...")
        processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
        print("Processor loaded.")
        
        print("✅ Success!")
        
    except Exception as e:
        print(f"❌ Error loading ColPali:")
        traceback.print_exc()

if __name__ == "__main__":
    test_colpali_loading()
