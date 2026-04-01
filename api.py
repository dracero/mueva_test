"""
============================================================================
HISTOLOGÍA RAG MULTIMODAL - API Backend
============================================================================
Backend FastAPI para servir el asistente RAG ultimodal a CopilotKit.
"""

import os
import uvicorn
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from muvera_test import AsistenteHistologiaMultimodal
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el asistente
asistente = AsistenteHistologiaMultimodal()

@app.on_event("startup")
async def startup_event():
    print("🚀 Iniciando backend y cargando modelos...")
    
    # Limpiar directorio de uploads al inicio para evitar "memoria" de imágenes anteriores
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        print(f"🧹 Limpiando directorio {uploads_dir}...")
        for file in uploads_dir.glob("*"):
            if file.is_file():
                file.unlink()
    
    asistente.inicializar_componentes()
    print("✅ Modelos cargados.")
    
    # Auto-indexación al inicio si hay PDFs y no hay colección
    try:
        pdf_dir = Path("./pdfs")
        if not pdf_dir.exists():
            pdf_dir = Path(".")
            
        pdfs = list(pdf_dir.glob("*.pdf"))
        if pdfs:
            print(f"📦 Se encontraron {len(pdfs)} PDFs. Verificando si es necesario indexar...")
            # Aquí podríamos verificar si la colección está vacía, pero por simplicidad
            # lanzamos el proceso. El método store_in_qdrant verifique si existe,
            # pero procesar_y_almacenar_pdfs_multimodal procesará todo de nuevo.
            # Idealmente, deberíamos chequear si la colección está vacía.
            
            # Chequeo rápido de colección vacía
            try:
                client = asistente.qdrant_client
                col_name = asistente.gestor_qdrant.content_mv_collection
                count = await client.count(col_name)
                if count.count == 0:
                    print("⚠️ Colección vacía. Iniciando indexación automática...")
                    await asistente.procesar_y_almacenar_pdfs_multimodal(pdfs, use_muvera=True)
                else:
                    print(f"✅ Colección {col_name} tiene {count.count} documentos. Saltando indexación.")
            except Exception:
                print("⚠️ Colección no encontrada o error. Iniciando indexación automática...")
                await asistente.procesar_y_almacenar_pdfs_multimodal(pdfs, use_muvera=True)
    except Exception as e:
        import traceback
        print(f"❌ Error en auto-indexación:\n{traceback.format_exc()}")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

# Wrapper para el agente
async def chat_handler(query: str, image_path: str = None, image_base64: str = None):
    """
    Manejador simple para conectar CopilotKit con el asistente.
    """
    print(f"📨 Consulta recibida: {query}")
    if image_path:
        print(f"🖼️ Imagen adjunta: {image_path}")
    if image_base64:
        print(f"🖼️ Imagen adjunta (Base64)")
    
    # Usar el flujo multimodal existente con imagen si está disponible
    resultado = await asistente.iniciar_flujo_multimodal(
        consulta_usuario=query,
        imagen_path=image_path,
        imagen_base64=image_base64
    )
    
    if resultado and resultado.get("respuesta"):
        return resultado["respuesta"]
    return "Lo siento, no pude generar una respuesta."

# Configurar CopilotKit
# sdk = CopilotKitSDK(
#     agents=[
#         LangGraphAgent(
#             name="histologia_agent",
#             description="Asistente de histología multimodal",
#             agent=chat_handler # Esto es un placeholder, idealmente adaptaremos el agente a LangGraph o usaremos un custom handler
#         )
#     ]
# )

# Como muvera_test.py no usa LangGraph nativamente, necesitamos un adaptador más simple.
# CopilotKit soporta "Simple Agent" o funciones directas.
# Vamos a exponer un endpoint personalizado para simplificar por ahora si la integración SDK es compleja sin LangGraph.

# REVISIÓN: CopilotKit Python SDK está muy orientado a LangGraph/LangChain.
# Dado que muvera_test usa una clase custom, lo mejor es crear un endpoint manual que simule la respuesta
# o adaptar la clase a LangChain Runnable.

from pydantic import BaseModel
from typing import Optional
import glob

class ChatRequest(BaseModel):
    messages: list
    image_path: Optional[str] = None  # Ruta de imagen opcional
    image_base64: Optional[str] = None # Imagen en base64

def get_latest_uploaded_image() -> Optional[str]:
    """Obtiene la imagen más reciente subida al servidor."""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        return None
    
    image_files = glob.glob(os.path.join(uploads_dir, "*"))
    if not image_files:
        return None
    
    # Ordenar por fecha de modificación (más reciente primero)
    image_files.sort(key=os.path.getmtime, reverse=True)
    return image_files[0] if image_files else None

@app.post("/copilotkit/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint manual para chat compatible con la estructura que espera un frontend simple,
    o para usar con useCopilotChat (que usa endpoints estándar).
    """
    last_message = request.messages[-1].get("content", "")
    
    # Buscar imagen: primero del request, luego la más reciente subida
    image_path = request.image_path
    image_base64 = request.image_base64
    
    if not image_path and not image_base64:
        image_path = get_latest_uploaded_image()
    
    if image_path and os.path.exists(image_path):
        print(f"🖼️ Usando imagen para contexto: {image_path}")
    elif image_base64:
        print(f"🖼️ Usando imagen Base64 para contexto")
    
    response_text = await chat_handler(last_message, image_path, image_base64)
    return {"response": response_text}

# Intento de uso estándar de CopilotKit si es posible envolver
# add_fastapi_endpoint(app, sdk, "/copilotkit")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Sube una imagen al servidor para análisis posterior.
    """
    try:
        # Guardar archivo localmente
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"🖼️ Imagen subida: {file_path}")
        return {"filename": file.filename, "path": file_path, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reindex")
async def reindex_pdfs():
    """
    Reindexa los PDFs en la carpeta ./pdfs
    IMPORTANTE: Borra las colecciones existentes para que los nuevos embeddings
    (con normalización L2) sean consistentes. Sin esto, se mezclarían vectores
    viejos (sin normalizar) con nuevos, produciendo scores incónsistentes.
    """
    try:
        print("\n🔄 Reindexando PDFs...")
        
        # 1. Borrar colecciones existentes para evitar mezcla de vectores
        client = asistente.gestor_qdrant.client
        for col in [
            asistente.gestor_qdrant.content_mv_collection,
            asistente.gestor_qdrant.content_fde_collection,
        ]:
            try:
                await client.delete_collection(col)
                print(f"   ✅ Colección borrada: {col}")
            except Exception:
                print(f"   ⚠️ No existía: {col}")
        
        # 2. Buscar PDFs en el directorio local ./pdfs/
        pdf_dir = Path("./pdfs")
        if pdf_dir.exists():
            archivos_existentes = list(pdf_dir.glob("*.pdf"))
        else:
            archivos_existentes = list(Path(".").glob("*.pdf"))
            
        if archivos_existentes:
            await asistente.procesar_y_almacenar_pdfs_multimodal(
                archivos_existentes,
                use_muvera=True
            )
            return {"status": "success", "message": f"Procesados {len(archivos_existentes)} archivos con embeddings normalizados"}
        return {"status": "warning", "message": "No se encontraron PDFs"}
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "detail": traceback.format_exc()}

if __name__ == "__main__":
    # Crear carpeta uploads si no existe
    os.makedirs("uploads", exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
