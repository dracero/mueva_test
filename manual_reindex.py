
import asyncio
import os
from pathlib import Path
from muvera_test import AsistenteHistologiaMultimodal
from dotenv import load_dotenv

load_dotenv()

async def manual_reindex():
    print("🚀 Starting Manual Reindex...")
    
    # 1. Initialize Assistant
    asistente = AsistenteHistologiaMultimodal()
    print("running inicializar_componentes...")
    asistente.inicializar_componentes()
    
    # 2. Find PDFs
    pdf_dir = Path("./pdfs")
    pdfs = []
    if pdf_dir.exists():
        pdfs = list(pdf_dir.glob("*.pdf"))
    else:
        pdfs = list(Path(".").glob("*.pdf"))
        
    print(f"📄 Found {len(pdfs)} PDFs: {[p.name for p in pdfs]}")
    
    if not pdfs:
        print("❌ No PDFs found!")
        return
        
    # 3. Process
    print("🔄 Calling procesar_y_almacenar_pdfs_multimodal...")
    try:
        await asistente.procesar_y_almacenar_pdfs_multimodal(pdfs, use_muvera=True)
        print("✅ Manual Reindex Completed.")
    except Exception as e:
        print(f"❌ Error during manual reindex: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(manual_reindex())
