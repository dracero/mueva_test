"""
Servidor principal FastAPI con arquitectura moderna y eficiente.
Basado en las mejores prácticas del repositorio de referencia.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Configurar encoding en Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar agente
from app.agent import HistologyAgent

# Inicializar agente globalmente
agent = HistologyAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor de ciclo de vida de la aplicación.
    Inicializa recursos al arrancar y limpia al detener.
    """
    logger.info("🚀 Iniciando servidor de histología...")
    
    # Limpiar directorio de uploads
    project_root = Path(__file__).resolve().parent.parent
    uploads_dir = project_root / "uploads"
    if uploads_dir.exists():
        logger.info(f"🧹 Limpiando directorio {uploads_dir}...")
        for file in uploads_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {file}: {e}")
    
    # Inicializar componentes del agente
    try:
        agent.inicializar_componentes()
        logger.info("✅ Componentes del agente inicializados")
    except Exception as e:
        logger.error(f"❌ Error inicializando agente: {e}", exc_info=True)
        raise
    
    # Esperar conexión con Qdrant
    logger.info("⏳ Verificando conexión con Qdrant...")
    max_retries = 15
    retry_interval = 2
    
    for i in range(max_retries):
        try:
            await agent.qdrant_client.get_collections()
            logger.info("✅ Conexión con Qdrant establecida")
            break
        except Exception as e:
            if i < max_retries - 1:
                logger.warning(
                    f"⚠️ Intento {i+1}/{max_retries}: Qdrant no responde. "
                    f"Reintentando en {retry_interval}s..."
                )
                import asyncio
                await asyncio.sleep(retry_interval)
            else:
                logger.error("❌ No se pudo conectar a Qdrant después de varios intentos")
                raise ConnectionError("Qdrant no está disponible")
    
    # Crear índices de payload
    try:
        from qdrant_client.models import PayloadSchemaType
        collections = [
            agent.gestor_qdrant.content_mv_collection,
            agent.gestor_qdrant.content_fde_collection
        ]
        
        for col in collections:
            try:
                await agent.qdrant_client.create_payload_index(
                    collection_name=col,
                    field_name="numero_pagina",
                    field_schema=PayloadSchemaType.INTEGER
                )
                await agent.qdrant_client.create_payload_index(
                    collection_name=col,
                    field_name="nombre_archivo",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info(f"✅ Índices creados en {col}")
            except Exception:
                pass  # Ya existen
    except Exception as e:
        logger.warning(f"⚠️ Error creando índices: {e}")
    
    # Auto-indexación de PDFs si la colección está vacía
    try:
        project_root = Path(__file__).resolve().parent.parent
        pdf_dir = project_root / "pdfs"
        if not pdf_dir.exists():
            # Buscar PDFs en directorio actual
            pdf_dir = project_root
        
        pdfs = list(pdf_dir.glob("*.pdf"))
        
        if pdfs:
            logger.info(f"📦 Encontrados {len(pdfs)} PDFs. Verificando indexación...")
            
            try:
                col_name = agent.gestor_qdrant.content_mv_collection
                count = await agent.qdrant_client.count(col_name)
                
                if count.count == 0:
                    logger.info("⚠️ Colección vacía. Iniciando indexación automática...")
                    await agent.procesar_pdfs([str(f) for f in pdfs])
                    logger.info("✅ Indexación completada")
                else:
                    logger.info(f"✅ Colección {col_name} tiene {count.count} documentos")
            except Exception:
                logger.info("⚠️ Colección no encontrada. Iniciando indexación...")
                await agent.procesar_pdfs([str(f) for f in pdfs])
                logger.info("✅ Indexación completada")
        else:
            logger.warning("⚠️ No se encontraron PDFs para indexar")
    except Exception as e:
        logger.error(f"❌ Error en auto-indexación: {e}", exc_info=True)
    
    logger.info("🎉 Servidor listo para recibir consultas")
    
    yield
    
    # Limpieza al detener
    logger.info("🛑 Deteniendo servidor...")
    try:
        agent.cerrar()
    except Exception as e:
        logger.error(f"Error cerrando agente: {e}")


# Crear aplicación FastAPI
app = FastAPI(
    title="Histología RAG Multimodal",
    description="Sistema RAG con ColPali + MUVERA para histopatología",
    version="2.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar directorio estático para imágenes
project_root = Path(__file__).resolve().parent.parent
os.makedirs(project_root / "histopatologia_data" / "embeddings", exist_ok=True)
app.mount(
    "/histopatologia_data",
    StaticFiles(directory=str(project_root / "histopatologia_data")),
    name="histopatologia_data"
)


# ============================================================================
# MODELOS DE DATOS
# ============================================================================

class ChatMessage(BaseModel):
    """Modelo de mensaje de chat"""
    content: str
    role: str = "user"


class ChatRequest(BaseModel):
    """Modelo de solicitud de chat"""
    messages: List[ChatMessage]
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


class ChatResponse(BaseModel):
    """Modelo de respuesta de chat"""
    response: str
    imagenes_recuperadas: List[dict] = []
    mostrar_imagenes: bool = False


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "histologia-rag",
        "version": "2.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal de chat.
    Procesa consultas de texto e imágenes.
    """
    try:
        # Extraer último mensaje
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        last_message = request.messages[-1].content
        logger.info(f"📨 Nueva consulta: {last_message[:100]}...")
        
        # Preparar imágenes si existen
        images = None
        if request.image_base64:
            images = [{
                "data": request.image_base64,
                "mime_type": "image/jpeg"
            }]
            logger.info("🖼️ Imagen base64 adjunta")
        elif request.image_path and os.path.exists(request.image_path):
            images = [{
                "data": request.image_path,
                "mime_type": "image/jpeg"
            }]
            logger.info(f"🖼️ Imagen desde archivo: {request.image_path}")
        
        # Procesar con streaming (pero recolectar resultado completo)
        final_response = None
        imagenes_relevantes = []
        
        async for chunk in agent.stream(last_message, "default_session", images):
            if chunk.get('is_task_complete'):
                final_response = chunk.get('content', '')
                imagenes_relevantes = chunk.get('imagenes_relevantes', [])
        
        # Determinar si mostrar imágenes
        tiene_imagen_adjunta = images is not None
        requiere_imagen = "mostr" in last_message.lower() or "ver imagen" in last_message.lower()
        mostrar_imagenes = requiere_imagen and not tiene_imagen_adjunta and len(imagenes_relevantes) > 0
        
        logger.info(f"✅ Respuesta generada: {len(final_response)} caracteres")
        logger.info(f"🖼️ Imágenes relevantes: {len(imagenes_relevantes)}")
        
        # Convertir rutas absolutas a rutas relativas para el frontend
        imagenes_limpias = []
        for img in (imagenes_relevantes or []):
            path_str = img.get("path", "")
            if "histopatologia_data" in path_str:
                idx = path_str.find("histopatologia_data")
                rel_path = path_str[idx:]
            else:
                rel_path = os.path.basename(path_str)
            imagenes_limpias.append({
                "path": rel_path,
                "descripcion": img.get("descripcion", "")
            })

        return ChatResponse(
            response=final_response or "No se pudo generar respuesta",
            imagenes_recuperadas=imagenes_limpias if mostrar_imagenes else [],
            mostrar_imagenes=mostrar_imagenes
        )
        
    except Exception as e:
        logger.error(f"❌ Error en chat: {e}", exc_info=True)
        
        # Detectar errores de cuota
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["quota", "resource_exhausted", "429", "rate limit"]):
            return ChatResponse(
                response="⚠️ Se agotó la cuota de la API. Por favor, esperá unos minutos.",
                imagenes_recuperadas=[],
                mostrar_imagenes=False
            )
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint para subir imágenes al servidor.
    Retorna la ruta del archivo guardado.
    """
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen"
            )
        
        # Guardar archivo
        project_root = Path(__file__).resolve().parent.parent
        uploads_dir = project_root / "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = str(uploads_dir / file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"🖼️ Imagen subida: {file_path}")
        
        return {
            "filename": file.filename,
            "path": file_path,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Error subiendo imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex_pdfs():
    """
    Reindexa todos los PDFs en el directorio ./pdfs
    
    IMPORTANTE: Borra las colecciones existentes para garantizar
    consistencia de los embeddings normalizados.
    """
    try:
        logger.info("🔄 Iniciando reindexación de PDFs...")
        
        # Borrar colecciones existentes
        collections = [
            agent.gestor_qdrant.content_mv_collection,
            agent.gestor_qdrant.content_fde_collection,
        ]
        
        for col in collections:
            try:
                await agent.qdrant_client.delete_collection(col)
                logger.info(f"✅ Colección borrada: {col}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo borrar {col}: {e}")
        
        # Buscar PDFs
        project_root = Path(__file__).resolve().parent.parent
        pdf_dir = project_root / "pdfs"
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
        else:
            pdfs = list(project_root.glob("*.pdf"))
        
        if not pdfs:
            return {
                "status": "warning",
                "message": "No se encontraron PDFs para indexar"
            }
        
        # Procesar PDFs
        logger.info(f"📄 Procesando {len(pdfs)} PDFs...")
        await agent.procesar_pdfs([str(f) for f in pdfs])
        
        logger.info("✅ Reindexación completada")
        
        return {
            "status": "success",
            "message": f"Reindexados {len(pdfs)} PDFs con embeddings normalizados"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en reindexación: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/stats")
async def get_stats():
    """
    Obtiene estadísticas del sistema.
    """
    try:
        # Obtener conteo de documentos
        mv_count = await agent.qdrant_client.count(
            agent.gestor_qdrant.content_mv_collection
        )
        fde_count = await agent.qdrant_client.count(
            agent.gestor_qdrant.content_fde_collection
        )
        
        return {
            "status": "ok",
            "collections": {
                "multi_vector": mv_count.count,
                "fde": fde_count.count
            },
            "model_info": agent.procesador.get_info() if agent.procesador else {}
        }
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Punto de entrada principal"""
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"🚀 Iniciando servidor en http://{host}:{port}")
    logger.info(f"📚 Documentación en http://{host}:{port}/docs")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
