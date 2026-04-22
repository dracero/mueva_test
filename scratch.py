import torch
from muvera_test import ProcesadorColPaliPuro
import numpy as np

async def main():
    procesador = ProcesadorColPaliPuro()
    import os, glob
    imgs = glob.glob("embeddings/*.jpg")
    if not imgs:
        print("No images found")
        return
        
    test_img = imgs[-1]
    print(f"Testing with {test_img}")
    
    emb1 = procesador.generar_embedding_imagen(test_img)
    emb2 = procesador.generar_embedding_imagen(test_img)
    
    sim_matrix = np.dot(emb1, emb2.T)
    maxsim = float(np.sum(np.max(sim_matrix, axis=1)))
    print(f"MaxSim exact: {maxsim}")
    
    # What if we save and load?
    from PIL import Image
    im = Image.open(test_img)
    im.save("test_resave.jpg", "JPEG", quality=80)
    emb3 = procesador.generar_embedding_imagen("test_resave.jpg")
    sim_matrix3 = np.dot(emb1, emb3.T)
    maxsim3 = float(np.sum(np.max(sim_matrix3, axis=1)))
    print(f"MaxSim resaved: {maxsim3}")

import asyncio
asyncio.run(main())
