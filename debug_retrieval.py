import asyncio
import os
import glob
from dotenv import load_dotenv
from muvera_test import SistemaRAGColPaliPuro, Config

# Cargar variables de entorno
load_dotenv()

async def debug_retrieval():
    print("🚀 Iniciando Debug de Recuperación con IMAGEN...")
    
    # Inicializar sistema
    sistema = SistemaRAGColPaliPuro()
    sistema.inicializar_componentes()
    
    # Buscar imagen subida más reciente
    uploads_dir = "uploads"
    list_of_files = glob.glob(f'{uploads_dir}/*')
    if not list_of_files:
        print("❌ No se encontraron imágenes en uploads/")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"🖼️ Usando imagen: {latest_file}")
    
    # Consulta multimodal
    consulta = "analiza esta imagen histologica y dime que organo es"
    
    print(f"\n\n🔎 CONSULTA MULTIMODAL: '{consulta}' + IMAGEN")
    
    # Ejecutar flujo completo y capturar logs (muvera_test ya imprime)
    respuesta = await sistema.procesar_consulta(consulta=consulta, imagen_path=latest_file)
    
    print("\n--------------------------------------------------")
    print("🤖 RESPUESTA FINAL DEL AGENTE:")
    print(respuesta)
    print("--------------------------------------------------")

    sistema.cerrar()

if __name__ == "__main__":
    asyncio.run(debug_retrieval())
