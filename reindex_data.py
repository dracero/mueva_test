import asyncio
import os
from muvera_test import AsistenteHistologiaMultimodal, limpiar_colecciones, Config

async def reindex():
    print("🔄 Iniciando re-indexación completa...")
    
    # Inicializar sistema
    asistente = AsistenteHistologiaMultimodal()
    asistente.inicializar_componentes()
    
    # Limpiar base de datos
    await limpiar_colecciones(asistente)
    
    # Buscar PDFs en directorio configurado
    pdf_dir = Config.BASE_DIR / "pdfs"
    archivos = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    if not archivos:
        print(f"⚠️ No se encontraron PDFs en {pdf_dir}")
        return

    print(f"📄 Procesando {len(archivos)} archivos: {archivos}")
    
    # Procesar de nuevo (esto usará la nueva lógica con metadatos de página)
    await asistente.procesar_pdfs(archivos, forzar=True)
    
    print("\n✅ Re-indexación completada con éxito.")
    asistente.cerrar()

if __name__ == "__main__":
    asyncio.run(reindex())
