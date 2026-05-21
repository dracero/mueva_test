import torch
from colpali_engine.models import ColPali
import traceback

def test():
    print("PyTorch CUDA available:", torch.cuda.is_available())
    x = torch.rand(2, 2).cuda()
    print("CUDA allocation successful:", x.device)
    
    print("Loading ColPali...")
    try:
        model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("Success")
    except Exception as e:
        traceback.print_exc()

test()
