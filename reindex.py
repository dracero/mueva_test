import asyncio
from muvera_test import AsistenteHistologiaMultimodal
from pathlib import Path
import os
import shutil

async def main():
    asistente = AsistenteHistologiaMultimodal()
    await asistente.inicializar_componentes()
    project_root = Path(__file__).resolve().parent
    print("Borrando embeddings locales...")
    embeddings_dir = project_root / "embeddings"
    if embeddings_dir.exists():
        shutil.rmtree(embeddings_dir)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    print("Borrando colecciones de Qdrant...")
    client = asistente.gestor_qdrant.client
    for col in [
        asistente.gestor_qdrant.content_mv_collection,
        asistente.gestor_qdrant.content_fde_collection,
    ]:
        try:
            await client.delete_collection(col)
            print(f"Colección borrada: {col}")
        except Exception:
            print(f"No existía: {col}")
            
    pdf_dir = project_root / "pdfs"
    if pdf_dir.exists():
        archivos_existentes = list(pdf_dir.glob("*.pdf"))
    else:
        archivos_existentes = list(project_root.glob("*.pdf"))
        
    if archivos_existentes:
        await asistente.procesar_y_almacenar_pdfs_multimodal(
            archivos_existentes,
            use_muvera=True
        )
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
