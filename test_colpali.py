import torch
from muvera_test import ProcesadorColPaliPuro
from PIL import Image
import asyncio

async def test():
    print("Iniciando ProcesadorColPaliPuro...")
    proc = ProcesadorColPaliPuro()
    print("Procesador iniciado. Generando embedding simulado...")
    try:
        # Create a dummy image
        img = Image.new('RGB', (224, 224), color = 'red')
        emb = proc.obtener_embeddings_imagenes([img])
        print("Embedding shape:", emb.shape)
        print("✅ SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
