import asyncio
import os
import sys
# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from muvera_test import AsistenteHistologiaMultimodal

async def main():
    asistente = AsistenteHistologiaMultimodal()
    asistente.inicializar_componentes()
    
    # Get the latest uploaded image
    import glob
    uploads_dir = "uploads"
    image_files = glob.glob(os.path.join(uploads_dir, "query_image_*"))
    image_files.sort(key=os.path.getmtime, reverse=True)
    if not image_files:
        print("No image found")
        return
    
    img_path = image_files[0]
    print(f"Testing with {img_path}")
    
    query_mv = asistente.procesador.generar_embedding_imagen(img_path)
    query_fde = asistente.procesador.generar_query_muvera(query_mv)
    
    print("Searching...")
    resultados, _ = await asistente.gestor_qdrant.buscar_muvera_2stage(
        query_mv, query_fde, top_k=20, min_score=0.0
    )
    
    for r in resultados:
        payload = r.get('payload', {})
        tipo = payload.get('tipo', '?')
        page = payload.get('numero_pagina', '?')
        print(f"Found: score={r['score']:.2f} | pág={page} | tipo={tipo}")

if __name__ == "__main__":
    asyncio.run(main())
