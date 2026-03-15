
# ============================================================================
# RAG HISTOPATOLOGÍA - LANGGRAPH + ColPali PURO + MUVERA
# Sistema simplificado usando SOLO ColPali para texto E imágenes
# ============================================================================
"""
🆕 Sistema RAG Multimodal Simplificado - SOLO ColPali + MUVERA + LangGraph

ARQUITECTURA SIMPLIFICADA:
- ColPali v1.2: Embeddings para TEXTO e IMÁGENES (UN SOLO MODELO)
- MUVERA: Two-stage retrieval (FDE rápido + MV preciso)
- Qdrant: Base de datos vectorial con soporte multi-vector
- LangGraph: Orquestación de agentes multi-paso
- Groq Llama-4 Scout: Generación de respuestas

VENTAJAS vs versión con ColBERT:
✅ Más simple (1 modelo en lugar de 2)
✅ Menos memoria GPU (~30% reducción)
✅ Consistencia total (mismo espacio de embeddings)
✅ ColPali maneja texto nativamente
✅ Código ~20% más corto
"""

import os
import re
import json
import time
import asyncio
import base64
import uuid
import nest_asyncio
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from pathlib import Path
import operator
import hashlib
import pickle
import gc
import gzip
import warnings
from io import BytesIO

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# PDFs e imágenes
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
try:
    import fitz # PyMuPDF
except ImportError:
    fitz = None
from PIL import Image, ImageEnhance
import numpy as np

# PyTorch
import torch

# Qdrant
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    MultiVectorConfig, MultiVectorComparator,
    Filter, HasIdCondition, FieldCondition
)

# MUVERA from fastembed
from fastembed.postprocess import Muvera

# ColPali - Visual document embeddings
from colpali_engine.models import ColPali as ColPaliModel
from colpali_engine.models import ColPaliProcessor

# LangChain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Groq para extracción de ontología
from groq import Groq as GroqClient

# Configuración de credenciales local
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

nest_asyncio.apply()
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración del sistema ColPali Puro + MUVERA"""

    # Adaptado para ejecución local
    BASE_DIR = Path("./histopatologia_data")
    EMBEDDINGS_DIR = BASE_DIR / "embeddings"
    ONTOLOGY_DIR = BASE_DIR / "ontologia"
    CACHE_DIR = BASE_DIR / "cache"

    ONTOLOGY_FILE = ONTOLOGY_DIR / "ontologia_histopatologia.json"

    # Dimensiones de embeddings (SOLO ColPali)
    COLPALI_EMBEDDING_DIM = 128  # ColPali dimensión por vector
    FDE_DIM = 20480              # MUVERA FDE dimension (64 clusters * 16 dim_proj * 20 reps)

    # Parámetros de procesamiento
    TEXT_CHUNK_SIZE = 1000
    TEXT_CHUNK_OVERLAP = 100
    IMAGE_DPI = 200
    MAX_IMAGE_SIZE = (1280, 1280)

    # Parámetros de memoria
    BATCH_SIZE = 8
    CLEAR_CACHE_AFTER_PROCESS = True

    # Mejoras visuales
    ENHANCE_CONTRAST = True
    ENHANCE_BRIGHTNESS = True
    CONTRAST_FACTOR = 1.2
    BRIGHTNESS_FACTOR = 1.1

    # Parámetros de búsqueda
    SEARCH_SCORE_THRESHOLD = 868.0
    SEARCH_PREFETCH_MULTIPLIER = 20  # Reducido de 50 para mejorar velocidad

    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.EMBEDDINGS_DIR, cls.ONTOLOGY_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

def setup_langsmith():
    """Configurar LangSmith para telemetría"""
    if not LANGSMITH_API_KEY:
        return False
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = "rag_histopatologia_llama_groq"
        print("✅ LangSmith configurado")
        return True
    except:
        print("⚠️ LangSmith no disponible")
        return False

def cleanup_memory():
    """Liberar memoria GPU/CPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ============================================================================
# EXTRACTOR DE ONTOLOGÍA
# ============================================================================

class ExtractorOntologia:
    """Extrae ontología histopatológica usando Groq"""

    def __init__(self, api_key: str):
        if not api_key:
            print("⚠️ API Key de Groq no proporcionada para ExtractorOntologia")
            self.model = None
            return
        self._groq_client = GroqClient(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def extraer_ontologia_completa(self, contenido: str, num_imagenes: int) -> Dict:
        """Extrae ontología completa del documento"""
        if not self.model:
            return {"sistemas_anatomicos": [], "metadata": {"tipo": "default"}}

        print(f"\n🔬 Extrayendo ontología de {len(contenido)} caracteres...")
        
        prompt = f"""Analiza este atlas de histopatología y extrae una ontología completa.

CONTENIDO TEXTUAL (muestra):
{contenido[:8000]}...

IMÁGENES: {num_imagenes} figuras

EXTRAE:
1. SISTEMAS ANATÓMICOS: órganos, tejidos, estructuras
2. TERMINOLOGÍA HISTOLÓGICA: tipos celulares, componentes tisulares
3. TÉCNICAS Y TINCIONES: métodos de procesamiento, coloraciones
4. FIGURAS: numeración y descripciones breves
5. PATOLOGÍAS: alteraciones, lesiones comunes

Responde SOLAMENTE con un JSON válido, sin texto adicional ni explicaciones."""

        for intento in range(2):
            try:
                prompt_actual = prompt if intento == 0 else \
                    f"Extrae una ontología en formato JSON puro (sin markdown) del siguiente texto de histopatología:\n{contenido[:5000]}"
                response = self._groq_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_actual}],
                    temperature=0
                )
                ontologia_texto = response.choices[0].message.content.strip()
                # Limpiar markdown code blocks
                if '```' in ontologia_texto:
                    # Extraer contenido entre los primeros ``` y los últimos ```
                    bloques = ontologia_texto.split('```')
                    for bloque in bloques:
                        bloque_limpio = bloque.strip()
                        if bloque_limpio.startswith('json'):
                            bloque_limpio = bloque_limpio[4:].strip()
                        if bloque_limpio.startswith('{'):
                            ontologia_texto = bloque_limpio
                            break
                
                ontologia = json.loads(ontologia_texto)

                ontologia["metadata"] = {
                    "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "modelo": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "num_imagenes": num_imagenes
                }

                with open(Config.ONTOLOGY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(ontologia, f, indent=2, ensure_ascii=False)

                print(f"✅ Ontología extraída: {len(ontologia)} categorías")
                return ontologia

            except json.JSONDecodeError as e:
                print(f"⚠️ Intento {intento+1}/2 - Error parsing JSON ontología: {e}")
                if intento == 0:
                    print("   Reintentando con prompt simplificado...")
                    continue
            except Exception as e:
                print(f"⚠️ Error ontología (intento {intento+1}): {e}")
                break

        print("⚠️ No se pudo extraer ontología. Continuando sin ella.")
        return {"sistemas_anatomicos": [], "metadata": {"tipo": "default"}}

    def cargar_ontologia(self) -> Optional[Dict]:
        """Cargar ontología desde archivo"""
        if Config.ONTOLOGY_FILE.exists():
            with open(Config.ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def buscar_en_ontologia(self, termino: str, ontologia: Dict) -> List[str]:
        """Buscar términos relevantes en ontología"""
        resultados = []
        termino_lower = termino.lower()

        def buscar_recursivo(obj, ruta=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    nueva_ruta = f"{ruta}/{k}" if ruta else k
                    if termino_lower in k.lower():
                        resultados.append(f"{nueva_ruta}: {str(v)[:100]}")
                    buscar_recursivo(v, nueva_ruta)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and termino_lower in item.lower():
                        resultados.append(f"{ruta}: {item}")

        buscar_recursivo(ontologia)
        return resultados[:5]

# ============================================================================
# PROCESADOR COLPALI PURO + MUVERA
# ============================================================================

class ProcesadorColPaliPuro:
    """
    Procesador simplificado usando SOLO ColPali para texto e imágenes
    """

    def __init__(self):
        print("\n🖼️ Inicializando ColPali Puro + MUVERA...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # SOLO ColPali - para texto E imágenes
        print("   📚 Cargando ColPali v1.2 (texto + imágenes)...")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            self.colpali_model = ColPaliModel.from_pretrained(
                "vidore/colpali-v1.2",
                quantization_config=quantization_config,
                device_map=self.device
            )
            self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            self.colpali_model.eval()
            print(f"   ✅ ColPali cargado ({Config.COLPALI_EMBEDDING_DIM}D multi-vector)")
        except Exception as e:
            print(f"   ⚠️ Error cargando ColPali: {e}")
            self.colpali_model = None
            self.colpali_processor = None

        # MUVERA configuration
        print("   🚀 Inicializando MUVERA...")
        self.muvera = Muvera(
            dim=128,        # ColPali embedding dimensionality
            k_sim=6,        # 64 clusters (2^6)
            dim_proj=16,    # Compress to 16 dimensions per cluster
            r_reps=20,      # 20 repetitions
            random_seed=42,
        )
        print(f"   ✅ MUVERA inicializado (FDE: {Config.FDE_DIM}D)")

    def __del__(self):
        """Liberar memoria al destruir objeto"""
        cleanup_memory()

    def extraer_imagenes_pdf(self, pdf_path: str) -> List[Dict]:
        """Extrae imágenes individuales del PDF (o páginas si no hay imágenes o falta PyMuPDF)"""
        print(f"📄 Extrayendo imágenes de {pdf_path}...")

        # Extraer imágenes del PDF
        imagenes = []
        nombre_base = Path(pdf_path).stem
        
        # Extracción detallada usando PyMuPDF con filtro estricto de tamaño
        # Dado que hay 1 figura principal por página, filtramos todo el ruido visual
        if fitz is not None:
            try:
                doc = fitz.open(pdf_path)
                image_count = 0
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_images_with_pos = []
                    valid_images_this_page = []
                    
                    # MÉTODO PRIMARIO: get_image_info(xrefs=True) devuelve SOLO las imágenes
                    # físicamente dibujadas en esta página (no el diccionario global del PDF)
                    img_info_list = page.get_image_info(xrefs=True)
                    page_xrefs = [info["xref"] for info in img_info_list if info.get("xref")]
                    page_y_positions = {info["xref"]: info.get("bbox", (0,0,0,0))[1] for info in img_info_list if info.get("xref")}
                    
                    if page_xrefs:
                        # Extraer imágenes por xref
                        for xref in page_xrefs:
                            try:
                                base_image = doc.extract_image(xref)
                                if not base_image:
                                    continue
                                
                                image_bytes = base_image["image"]
                                ext = base_image["ext"]
                                y_position = page_y_positions.get(xref, 0.0)
                                
                                image_count += 1
                                img_path = Config.EMBEDDINGS_DIR / f"{nombre_base}_p{page_num+1}_img{image_count}.{ext}"
                                with open(img_path, "wb") as f:
                                    f.write(image_bytes)
                                    
                                img_pil = Image.open(img_path)
                                width, height = img_pil.size
                                area = width * height
                                
                                # Filtro mínimo bajo: solo descarta íconos diminutos
                                if width >= 150 and height >= 150:
                                    if img_pil.mode != "RGB":
                                        img_pil = img_pil.convert("RGB")
                                    
                                    img_path_rgb = Config.EMBEDDINGS_DIR / f"{nombre_base}_p{page_num+1}_img{image_count}.jpg"
                                    img_pil.save(img_path_rgb, "JPEG")
                                    
                                    if str(img_path) != str(img_path_rgb):
                                        os.remove(img_path)
                                        
                                    img_path = img_path_rgb

                                    valid_images_this_page.append({
                                        "page": page_num + 1,
                                        "path": str(img_path),
                                        "type": "extracted_figure",
                                        "size": (width, height),
                                        "area": area,
                                        "y_position": y_position
                                    })
                                else:
                                    os.remove(img_path)
                            except Exception as e:
                                print(f"⚠️ Error procesando xref {xref} en página {page_num+1}: {e}")
                    else:
                        # FALLBACK: get_image_info vacío → la imagen está incrustada como Form XObject
                        # Renderizar la página completa como imagen usando pdf2image
                        try:
                            page_imgs = convert_from_path(
                                pdf_path, first_page=page_num+1, last_page=page_num+1,
                                dpi=Config.IMAGE_DPI, fmt='jpeg', size=Config.MAX_IMAGE_SIZE
                            )
                            if page_imgs:
                                image_count += 1
                                img_path = Config.EMBEDDINGS_DIR / f"{nombre_base}_p{page_num+1}_img{image_count}.jpg"
                                page_imgs[0].save(str(img_path), "JPEG")
                                width, height = page_imgs[0].size
                                valid_images_this_page.append({
                                    "page": page_num + 1,
                                    "path": str(img_path),
                                    "type": "page_render",
                                    "size": (width, height),
                                    "area": width * height,
                                    "y_position": 0.0
                                })
                                print(f"      📸 Pg {page_num+1}: renderizada como página completa ({width}x{height})")
                        except Exception as e:
                            print(f"⚠️ Error renderizando página {page_num+1}: {e}")
                    
                    # Conservar SOLO la imagen más grande de la página
                    if valid_images_this_page:
                        largest_image = max(valid_images_this_page, key=lambda x: x["area"])
                        
                        for img_data in valid_images_this_page:
                            if img_data["path"] != largest_image["path"]:
                                try:
                                    os.remove(img_data["path"])
                                except OSError:
                                    pass
                                    
                        page_images_with_pos.append(largest_image)
                    
                    for idx, img_data in enumerate(page_images_with_pos):
                        img_data["img_index_in_page"] = idx
                        img_data["total_images_in_page"] = len(page_images_with_pos)
                    
                    imagenes.extend(page_images_with_pos)

                if len(imagenes) > 0:
                    print(f"✅ {len(imagenes)} figuras extraídas vía PyMuPDF + fallback de renderizado")
                    return imagenes
                else:
                    print("⚠️ No se encontraron figuras grandes, recurriendo a renderizado de página completa.")
            except Exception as e:
                print(f"⚠️ Error intentando extracción con PyMuPDF: {e}. Usando fallback a página entera.")
        else:
            print("⚠️ PyMuPDF (fitz) no instalado. Usando extracción de página completa.")
            
        # Fallback a renderizado de página completa
        try:
            # Nota: Requiere Poppler instalado en el sistema
            pages = convert_from_path(
                pdf_path,
                dpi=Config.IMAGE_DPI,
                fmt='jpeg',
                size=Config.MAX_IMAGE_SIZE
            )
            
            for page_num, page_image in enumerate(pages, 1):
                img_path = Config.EMBEDDINGS_DIR / f"{nombre_base}_page_{page_num}.jpg"
                page_image.save(img_path, quality=90, optimize=True)

                imagenes.append({
                    "page": page_num,
                    "path": str(img_path),
                    "type": "full_page",
                    "size": page_image.size
                })

                del page_image

            print(f"✅ {len(imagenes)} páginas extraídas")
            return imagenes

        except Exception as e:
            print(f"❌ Error extrayendo imágenes: {e}")
            return []

    def _preprocesar_imagen(self, imagen_path: str) -> Image.Image:
        """Preprocesamiento específico para histopatología"""
        image = Image.open(imagen_path).convert("RGB")

        if Config.ENHANCE_CONTRAST:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(Config.CONTRAST_FACTOR)

        if Config.ENHANCE_BRIGHTNESS:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(Config.BRIGHTNESS_FACTOR)

        return image

    def generar_embedding_imagen(self, imagen_path: str) -> Optional[np.ndarray]:
        """
        Genera embedding ColPali multi-vector para imagen
        """
        if self.colpali_model is None:
            print("⚠️ ColPali no disponible")
            return None

        try:
            image = self._preprocesar_imagen(imagen_path)
            
            batch_images = self.colpali_processor.process_images([image])
            batch_images = {k: v.to(self.colpali_model.device) for k, v in batch_images.items()}

            with torch.no_grad():
                image_embeddings = self.colpali_model(**batch_images)

            # Multi-vector output (late interaction)
            multivector = image_embeddings[0].cpu().float().numpy()

            del image, batch_images, image_embeddings
            # cleanup_memory()  <-- Removido por lentitud excesiva

            return multivector

        except Exception as e:
            print(f"❌ Error generando embedding imagen: {e}")
            return None

    def generar_embedding_texto(self, texto: str) -> Optional[np.ndarray]:
        """
        Genera embedding ColPali multi-vector para TEXTO
        """
        if self.colpali_model is None:
            print("⚠️ ColPali no disponible")
            return None

        try:
            # ColPali procesa queries textuales
            batch_queries = self.colpali_processor.process_queries([texto])
            batch_queries = {k: v.to(self.colpali_model.device) for k, v in batch_queries.items()}
            
            with torch.no_grad():
                text_embeddings = self.colpali_model(**batch_queries)
            
            # Multi-vector output
            multivector = text_embeddings[0].cpu().float().numpy()
            
            del batch_queries, text_embeddings
            # cleanup_memory()  <-- Removido por lentitud excesiva
            
            return multivector

        except Exception as e:
            print(f"❌ Error generando embedding texto: {e}")
            return None

    def generar_fde_muvera(self, multivectors: np.ndarray) -> np.ndarray:
        """
        Genera Fixed Dimensional Encoding (FDE) usando MUVERA
        """
        mv = np.array(multivectors, dtype=np.float32)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)

        fde = self.muvera.process_document(mv)
        return fde

    def generar_query_muvera(self, query_multivectors: np.ndarray) -> np.ndarray:
        """Procesar query con MUVERA para búsqueda en colección FDE"""
        mv = np.array(query_multivectors, dtype=np.float32)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)
        return self.muvera.process_query(mv)

    def get_info(self) -> Dict[str, Any]:
        """Retorna información sobre el procesador"""
        return {
            "modelo": "ColPali v1.2 (PURO - texto + imágenes)",
            "muvera": "Activo",
            "device": self.device,
            "embedding_dim": Config.COLPALI_EMBEDDING_DIM,
            "fde_dim": Config.FDE_DIM
        }

# ============================================================================
# GESTOR DE QDRANT CON MUVERA
# ============================================================================

class GestorQdrantMuvera:
    """
    Gestor de Qdrant con arquitectura dual MUVERA
    """

    def __init__(self, url: str, api_key: str, collection_base: str = "histopatologia"):
        self.url = url
        self.api_key = api_key
        self.collection_base = collection_base

        self._client = None

        # Nombres de colecciones
        self.content_mv_collection = f"{collection_base}_content_mv"
        self.content_fde_collection = f"{collection_base}_content_fde"

    @property
    def client(self):
        """Cliente Qdrant cacheado"""
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=120,
                prefer_grpc=False
            )
            print("🔗 Cliente Qdrant conectado")
        return self._client

    async def crear_colecciones(self):
        """Crear colecciones multi-vector y FDE"""
        print("\n📦 Creando colecciones Qdrant...")
        
        client = self.client
        
        # Colección multi-vector (para reranking)
        try:
            await client.get_collection(self.content_mv_collection)
        except:
            await client.create_collection(
                collection_name=self.content_mv_collection,
                vectors_config=VectorParams(
                    size=Config.COLPALI_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    )
                )
            )

        # Colección FDE (para fast search)
        try:
            await client.get_collection(self.content_fde_collection)
        except:
            await client.create_collection(
                collection_name=self.content_fde_collection,
                vectors_config=VectorParams(
                    size=Config.FDE_DIM,
                    distance=Distance.COSINE
                )
            )

        print("✅ Colecciones listas")

    async def insertar_batch_muvera(
        self,
        points_mv: List[PointStruct],
        points_fde: List[PointStruct]
    ):
        """
        Inserta batch con arquitectura dual MUVERA
        """
        client = self.client
        await client.upsert(collection_name=self.content_mv_collection, points=points_mv, wait=True)
        await client.upsert(collection_name=self.content_fde_collection, points=points_fde, wait=True)

    async def buscar_muvera_2stage(
        self,
        query_multivector: np.ndarray,
        query_fde: np.ndarray,
        top_k: int = 5,
        prefetch_multiplier: int = Config.SEARCH_PREFETCH_MULTIPLIER,
        min_score: float = 0.0,
        figuras_filtro: List[str] = None
    ) -> Tuple[List[Dict], bool]:
        """
        Búsqueda 2-stage con MUVERA. Retorna (resultados, has_rejected_candidates)
        """
        client = self.client
        has_rejected = False

        try:
            # STAGE 1: Fast FDE search
            # Optimizacion: with_payload=False para ahorrar memoria/ancho de banda
            qdrant_filter = None
            if figuras_filtro:
                from qdrant_client.models import MatchAny
                qdrant_filter = Filter(
                    should=[
                        FieldCondition(
                            key="figuras",
                            match=MatchAny(any=figuras_filtro)
                        )
                    ]
                )

            fde_response = await client.query_points(
                collection_name=self.content_fde_collection,
                query=query_fde.tolist(),
                query_filter=qdrant_filter,
                limit=top_k * prefetch_multiplier,
                with_payload=False
            )
            
            if not fde_response.points:
                return [], False
            
            candidate_ids = [point.id for point in fde_response.points]

            # STAGE 2: Precise multi-vector reranking
            mv_response = await client.query_points(
                collection_name=self.content_mv_collection,
                query=query_multivector.tolist(),
                query_filter=Filter(
                    must=[HasIdCondition(has_id=candidate_ids)]
                ),
                limit=top_k * 2  # Traemos un poco más para filtrar después
            )

            resultados = []
            descartados = 0
            print(f"   🕵️ Reranking {len(mv_response.points)} candidatos (Threshold: {min_score})...")
            for r in mv_response.points:
                score = float(r.score)
                # Si estamos buscando específicamente una figura y esta la tiene, la guardamos sin importar el score
                tiene_figura_exacta = False
                if figuras_filtro and "figuras" in r.payload:
                    if any(f in r.payload["figuras"] for f in figuras_filtro):
                        tiene_figura_exacta = True
                        
                if score >= min_score or tiene_figura_exacta:
                    resultados.append({
                        "id": r.id,
                        "score": score,
                        "payload": r.payload
                    })
                else:
                    descartados += 1

            if descartados > 0:
                print(f"      🗑️ {descartados} candidatos descartados (score < {min_score})")

            # Ordenamos por score descendente por si acaso (aunque Qdrant ya los trae ordenados)
            resultados.sort(key=lambda x: x['score'], reverse=True)
            
            # Limitar a top_k *después* de filtrar
            resultados = resultados[:top_k]
            
            # Si NO hay resultados válidos después de filtrar, entonces consideramos "has_rejected" = True
            if not resultados:
                has_rejected = True
                
            return resultados, has_rejected

        except Exception as e:
            print(f"❌ Error búsqueda MUVERA: {e}")
            return [], False

# ============================================================================
# GESTOR DE MEMORIA A LARGO PLAZO (CHROMADB)
# ============================================================================

import sqlite3
import datetime

class MemoriaSQLite:
    """Gestor de memoria lineal usando SQLite y resúmenes con LLM"""
    def __init__(self, db_path: str = "./chat_memory.sqlite"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT,
                summary TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def add_interaction_summary(self, session_id: str, user_query: str, summary: str):
        if not user_query.strip() or not summary.strip():
            return
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO interactions (session_id, timestamp, user_query, summary) VALUES (?, ?, ?, ?)",
                (session_id, datetime.datetime.now(), user_query, summary)
            )
            conn.commit()
            conn.close()
            print(f"   💾 Resumen guardado en memoria SQLite (Sesión: {session_id})")
        except Exception as e:
            print(f"   ⚠️ Error guardando en SQLite: {e}")
            
    def get_relevant_history(self, session_id: str, n_results: int = 5) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            # Obtener las interacciones más recientes para dar contexto lineal temporal
            c.execute(
                "SELECT summary FROM interactions WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", 
                (session_id, n_results)
            )
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                return ""
            
            # Invertimos para que queden en orden cronológico (más viejo a más nuevo)
            rows.reverse()
            history = "\n---\n".join([row[0] for row in rows])
            return history
        except Exception as e:
            print(f"   ⚠️ Error recuperando memoria SQLite: {e}")
            return ""

# ============================================================================
# ESTADO DEL GRAFO LANGGRAPH
# ============================================================================

class AgentState(TypedDict):
    """Estado del sistema de agentes"""
    messages: Annotated[list, add_messages]
    consulta_usuario: str
    imagen_consulta: Optional[str]
    contexto_memoria: str
    ontologia: Dict
    contexto_ontologico: str
    clasificacion: str
    consulta_optimizada: str
    filtros_ontologia: List[str]
    resultados_busqueda: List[Dict[str, Any]]
    contexto_documentos: str
    imagenes_relevantes: List[str]
    respuesta_final: str
    trayectoria: Annotated[List[Dict[str, Any]], operator.add]
    imagen_base64: Optional[str]
    user_id: str

    tiempo_inicio: float
    abortar_reset: bool

# ============================================================================
# SISTEMA PRINCIPAL CON LANGGRAPH
# ============================================================================

class SistemaRAGColPaliPuro:
    """
    Sistema RAG Multimodal SIMPLIFICADO con LangGraph y ColPali PURO
    """

    def __init__(self):
        Config.setup_directories()
        self._setup_apis()

        self.llm = None
        self.procesador = None
        self.gestor_qdrant = None
        self.extractor_ontologia = None
        self.ontologia = None
        self.compiled_graph = None
        self.compiled_graph = None
        self.memoria = None # Se inicializa despues del procesador

    def _setup_apis(self):
        """Configurar APIs"""
        if GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        setup_langsmith()

    def inicializar_componentes(self):
        """Inicializar todos los componentes"""
        print("\n" + "="*80)
        print("🚀 SISTEMA RAG HISTOPATOLOGÍA - ColPali PURO + MUVERA + LangGraph")
        print("="*80)

        # LLM
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            api_key=GROQ_API_KEY
        )

        # Configurar directorio de uploads para imágenes temporales
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Procesador ColPali PURO
        self.procesador = ProcesadorColPaliPuro()

        # Memoria SQLite Lineal
        self.memoria = MemoriaSQLite(db_path=str(Config.BASE_DIR / 'chat_memory.sqlite'))

        # Qdrant
        self.gestor_qdrant = GestorQdrantMuvera(
            url=QDRANT_URL or "http://localhost:6333",
            api_key=QDRANT_KEY,
            collection_base="histopatologia"
        )

        # Extractor de ontología
        self.extractor_ontologia = ExtractorOntologia(GROQ_API_KEY)
        self.ontologia = self.extractor_ontologia.cargar_ontologia()

        # LangGraph
        self._inicializar_langgraph()
        cleanup_memory()

    def _inicializar_langgraph(self):
        """Inicializar grafo de agentes"""
        graph = StateGraph(AgentState)

        graph.add_node("recepcionar_consulta", self._nodo_recepcionar_consulta)
        graph.add_node("inicializar", self._nodo_inicializar)
        graph.add_node("analizar_ontologia", self._nodo_analizar_ontologia)
        graph.add_node("clasificar", self._nodo_clasificar)
        graph.add_node("optimizar_consulta", self._nodo_optimizar_consulta)
        graph.add_node("buscar", self._nodo_buscar)
        graph.add_node("generar_respuesta", self._nodo_generar_respuesta)
        graph.add_node("reset", self._nodo_reset)
        graph.add_node("finalizar", self._nodo_finalizar)

        graph.add_edge(START, "recepcionar_consulta")
        graph.add_edge("recepcionar_consulta", "inicializar")
        graph.add_edge("inicializar", "analizar_ontologia")
        graph.add_edge("analizar_ontologia", "clasificar")
        graph.add_edge("clasificar", "optimizar_consulta")
        graph.add_edge("optimizar_consulta", "buscar")
        
        # Condicional después de buscar
        graph.add_conditional_edges(
            "buscar",
            self._decidir_camino_tras_busqueda,
            {
                "generar": "generar_respuesta",
                "reset": "reset"
            }
        )
        
        graph.add_edge("generar_respuesta", "finalizar")
        graph.add_edge("reset", "finalizar")
        graph.add_edge("finalizar", END)

        self.compiled_graph = graph.compile()

    # ========== NODOS DEL GRAFO ==========

    async def _nodo_recepcionar_consulta(self, state: AgentState) -> AgentState:
        """Nodo 0: Recepcionar consulta y procesar imagen Base64 si existe"""
        print(f"\n📨 Recibiendo consulta: {state['consulta_usuario'][:50]}...")
        
        state["trayectoria"] = [{"nodo": "recepcionar_consulta", "timestamp": time.time()}]
        
        # Procesar imagen Base64 si existe
        if state.get("imagen_base64"):
            try:
                print("🖼️ Procesando imagen Base64...")
                # Decodificar base64
                image_data = base64.b64decode(state["imagen_base64"])
                
                # Generar nombre único
                filename = f"query_image_{uuid.uuid4().hex}.jpg"
                filepath = self.uploads_dir / filename
                
                # Guardar imagen
                with open(filepath, "wb") as f:
                    f.write(image_data)
                
                state["imagen_consulta"] = str(filepath)
                print(f"✅ Imagen guardada en: {filepath}")
                
            except Exception as e:
                print(f"❌ Error decodificando imagen Base64: {e}")
                # No fallamos, solo continuamos sin imagen
                state["imagen_consulta"] = None
        
        return state

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        state["ontologia"] = self.ontologia or {}
        state["tiempo_inicio"] = time.time()
        
        print(f"   🧠 Recuperando memoria semántica para sesión {state['user_id']}...")
        history = self.memoria.get_relevant_history(
            session_id=state["user_id"],
            n_results=5
        )
        state["contexto_memoria"] = history
        if history:
            print(f"   ✅ Memoria recuperada: {len(history)} caracteres")
            
        state["trayectoria"].append({"nodo": "inicializar", "timestamp": time.time()})
        return state

    async def _nodo_analizar_ontologia(self, state: AgentState) -> AgentState:
        if not state["ontologia"]:
            state["contexto_ontologico"] = "No disponible"
            state["filtros_ontologia"] = []
        else:
            terminos = self.extractor_ontologia.buscar_en_ontologia(state["consulta_usuario"], state["ontologia"])
            state["contexto_ontologico"] = "\n".join(terminos)
            state["filtros_ontologia"] = [t.split(":")[1].strip() for t in terminos[:3]] if terminos else []

        state["trayectoria"].append({"nodo": "analizar_ontologia", "timestamp": time.time()})
        return state

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        info_imagen = f"\nImagen adjunta: Sí" if state.get('imagen_consulta') else "\nImagen adjunta: No"
        messages = [
            SystemMessage(content="Eres un experto en histopatología. Clasifica consultas."),
            HumanMessage(content=f"CONSULTA: {state['consulta_usuario']}{info_imagen}\nCONTEXTO ONTOLÓGICO:\n{state['contexto_ontologico']}")
        ]
        response = await self.llm.ainvoke(messages)
        state["clasificacion"] = response.content
        state["trayectoria"].append({"nodo": "clasificar", "timestamp": time.time()})
        return state

    async def _nodo_optimizar_consulta(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content="Optimiza consultas para búsqueda RAG multimodal."),
            HumanMessage(content=f"CONSULTA ORIGINAL: {state['consulta_usuario']}\nCONTEXTO ONTOLÓGICO: {state['contexto_ontologico'][:500]}")
        ]
        response = await self.llm.ainvoke(messages)
        state["consulta_optimizada"] = response.content
        state["trayectoria"].append({"nodo": "optimizar_consulta", "timestamp": time.time()})
        return state

    async def _nodo_buscar(self, state: AgentState) -> AgentState:
        resultados = []
        has_rejected = False
        state["abortar_reset"] = False # Default
        if state.get('imagen_consulta') and os.path.exists(state['imagen_consulta']):
            query_mv = self.procesador.generar_embedding_imagen(state['imagen_consulta'])
        else:
            query_mv = self.procesador.generar_embedding_texto(state['consulta_optimizada'])

        if query_mv is not None:
            t0 = time.time()
            print(f"\n🔍 Ejecutando búsqueda en Qdrant...")
            
            # Generar query FDE (usando método correcto para queries)
            query_fde = self.procesador.generar_query_muvera(query_mv)
            
            # Buscar menciones de figuras en la consulta para forzar el filtrado exacto si es posible
            figuras_en_consulta = self._extraer_figuras_de_texto(state['consulta_optimizada'])
            
            t1 = time.time()
            resultados, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                query_mv, 
                query_fde,
                min_score=Config.SEARCH_SCORE_THRESHOLD,
                figuras_filtro=figuras_en_consulta
            )
            t2 = time.time()
            print(f"⏱️ Tiempos: FDE={(t1-t0):.2f}s | Búsqueda+Rerank={(t2-t1):.2f}s")
            
            # Filtrar resultados: solo mantener la PRIMERA imagen (mayor score) + todos los textos
            primera_imagen_vista = False
            resultados_filtrados = []
            for r in resultados:
                tipo = r.get('payload', {}).get('tipo', 'unknown')
                if tipo == 'imagen':
                    if not primera_imagen_vista:
                        primera_imagen_vista = True
                        resultados_filtrados.append(r)
                    else:
                        img_name = os.path.basename(r.get('payload', {}).get('imagen_path', 'N/A'))
                        print(f"   ⏭️ Imagen adicional descartada (solo se usa la principal): {img_name}")
                else:
                    resultados_filtrados.append(r)
            resultados = resultados_filtrados
            
            print(f"\n📄 Resultados recuperados ({len(resultados)}):")
            for i, res in enumerate(resultados):
                payload = res.get('payload', {})
                score = res.get('score', 0.0)
                doc_name = payload.get('nombre_archivo', 'unknown')
                page_num = payload.get('numero_pagina', '?')
                print(f"   [{i+1}] Score: {score:.4f} | Doc: {doc_name} (Pg {page_num})")
                if payload.get('tipo') == 'texto':
                     print(f"       Texto: {payload.get('texto', '')[:100]}...")
                elif payload.get('tipo') == 'imagen':
                     print(f"       Imagen: {payload.get('imagen_path', 'N/A')}")

        state["resultados_busqueda"] = resultados
        state["abortar_reset"] = has_rejected

        if has_rejected:
            print("🚨 ALERTA: Candidatos rechazados detectados. Se abortará la generación para evitar errores de contexto excesivo.")
            contextos = []
            state["contexto_documentos"] = ""
            state["imagenes_relevantes"] = []
        else:
            contextos = []
            imagenes = []
            for i, r in enumerate(resultados):
                score = r.get('score', 0.0)
                tipo = r['payload'].get('tipo', 'unknown')
                pdf_name = r['payload'].get('pdf_name', 'desconocido')
                page_num = r['payload'].get('numero_pagina', '?')
                
                if tipo == 'texto':
                    texto = r['payload'].get('texto', '')
                    figuras = r['payload'].get('figuras', [])
                    figuras_str = ", ".join(figuras) if figuras else ""
                    figuras_info = f"\nFiguras mencionadas: {figuras_str}" if figuras_str else ""
                    contextos.append(f"[RESULTADO {i+1} - TEXTO - Score: {score:.2f} - Fuente: {pdf_name} (Pg {page_num})]{figuras_info}\n{texto[:800]}")
                elif tipo == 'imagen':
                    img_path = r['payload'].get('imagen_path')
                    contexto_texto = r['payload'].get('contexto_texto', '')
                    figuras = r['payload'].get('figuras', [])
                    figuras_str = ", ".join(figuras) if figuras else "No identificadas"
                    if img_path:
                        imagenes.append(img_path)
                        contextos.append(f"[RESULTADO PRINCIPAL - IMAGEN - Score: {score:.2f} - Fuente: {pdf_name} (Pg {page_num})]\nArchivo: {os.path.basename(img_path)}\nFiguras en esta página: {figuras_str}\nTexto asociado a esta imagen: {contexto_texto[:600]}")

            state["contexto_documentos"] = "\n\n---\n\n".join(contextos)
            state["imagenes_relevantes"] = imagenes
            
        state["trayectoria"].append({"nodo": "buscar", "timestamp": time.time()})
        return state

    def _decidir_camino_tras_busqueda(self, state: AgentState) -> str:
        """Decide si ir a generar respuesta o resetear"""
        if state.get("abortar_reset", False):
            return "reset"
        return "generar"

    async def _nodo_reset(self, state: AgentState) -> AgentState:
        """Nodo de reset para detener generación insegura"""
        print("🛑 RESET SYSTEM triggered due to low confidence candidates.")
        state["respuesta_final"] = "No esta en base de datos."
        state["imagenes_relevantes"] = []
        state["contexto_documentos"] = ""
        # Aquí podríamos limpiar más cosas si fuera necesario
        state["trayectoria"].append({"nodo": "reset", "timestamp": time.time()})
        return state

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        """Nodo 6: Generar respuesta basada EXCLUSIVAMENTE en contexto recuperado"""
        print("\n💭 Generando respuesta basada en contexto recuperado...")

        # Construir información sobre imágenes
        info_imagen = ""
        if state.get('imagen_consulta'):
            info_imagen = "\nNOTA: El usuario proporcionó una imagen para análisis. Las imágenes recuperadas de la base de datos son las MEJORES COINCIDENCIAS encontradas para esa imagen."
        if state["imagenes_relevantes"]:
            info_imagen += f"\nSe encontraron {len(state['imagenes_relevantes'])} imágenes coincidentes en la base de datos. DEBES describir estas imágenes usando el texto asociado."

        # Cargar imágenes recuperadas
        content_parts = []
        
        # 1. Instrucciones del sistema (modificadas para multimodal)
        system_prompt = """Eres un profesor experto en histopatología. Tu función es responder usando la información textual Y VISUAL recuperada de la base de datos.

REGLA FUNDAMENTAL SOBRE IMÁGENES RECUPERADAS:
Las imágenes etiquetadas como [IMAGEN RECUPERADA] son el RESULTADO de una búsqueda por similitud en la base de datos. Estas imágenes YA PASARON un umbral de similitud alto y SON la mejor coincidencia encontrada. Por lo tanto:
- DEBES describir y analizar las imágenes recuperadas usando el texto asociado que se proporciona en el contexto.
- Si el usuario subió una imagen (etiquetada como [IMAGEN DE CONSULTA DEL USUARIO]), la imagen recuperada ES la coincidencia encontrada para esa consulta. Úsala como fuente de verdad.
- NO rechaces una imagen recuperada diciendo que "no coincide" con la del usuario. El sistema de búsqueda ya validó la coincidencia.
- Si una página contiene múltiples figuras (ej: Figura 11-2 y 11-3), DEBES analizar visualmente la imagen recuperada y compararla con las DESCRIPCIONES de cada figura en el texto asociado para determinar cuál es. NO asumas que es la primera figura de la lista. Identifica la figura correcta basándote en las características visuales (tipo de tejido, tinción, estructuras visibles) y las descripciones del texto.
- Si el usuario preguntó por una figura específica, ENFÓCATE SOLO en esa figura y su descripción del texto asociado.

REGLAS DE PRECISIÓN:
1. Responde basándote en el contexto proporcionado (imágenes recuperadas y fragmentos de texto).
2. Puedes realizar deducciones lógicas apoyadas en el texto o imágenes del contexto, citando qué parte te permite deducirlo.
3. Si NO se recuperó NINGUNA imagen ni texto relevante (contexto vacío), responde EXACTAMENTE: "No hay suficiente contexto o detalle en las fuentes proporcionadas para dar una respuesta precisa a tu consulta."
4. NUNCA inventes información que no esté en el contexto.

REGLAS SOBRE FIGURAS:
1. Cada resultado indica qué figuras contiene esa página (campo "Figuras en esta página" o "Figuras mencionadas"). USA esta información.
2. Si el usuario pregunta por una figura específica (ej: "Figura 14.3"), SOLO describe esa figura usando el texto asociado. NO describas las otras figuras de la misma página.
3. Si la imagen recuperada contiene múltiples figuras en una sola imagen, identifica y describe SOLO la que el usuario solicitó.

ESTRUCTURA DE RESPUESTA:
1. **Imagen encontrada**: Indica qué imagen se recuperó de la base de datos y su figura correspondiente.
2. **Análisis Visual**: Describe qué se observa en la imagen recuperada según el texto asociado.
3. **Identificación**: Qué órgano/tejido/estructura se observa.
4. **Evidencia**: Integra lo que se ve en la imagen con lo que dice el texto del contexto."""

        messages = [
            SystemMessage(content=system_prompt)
        ]

        if state.get("contexto_memoria"):
            historial = f"\n========================================\nHISTORIAL DE CONVERSACIÓN RELEVANTE:\n{state['contexto_memoria']}\n========================================\n"
        else:
            historial = ""

        # 2. Construir mensaje de usuario con texto e imágenes
        user_content = [
            {"type": "text", "text": f"""{historial}CONSULTA DEL USUARIO: {state["consulta_usuario"]}
{info_imagen}

========================================
CONTEXTO RECUPERADO DE LA BASE DE DATOS
(Esta es la ÚNICA fuente de verdad para tu respuesta)
========================================

{state["contexto_documentos"][:10000]}

========================================

Responde basándote ÚNICAMENTE en el contexto de arriba y las IMÁGENES adjuntas.
"""}
        ]

        # Limitar número de imágenes por restricción de Groq (max 5)
        max_imagenes_recuperadas = 4 if (state.get('imagen_consulta') and os.path.exists(state['imagen_consulta'])) else 5
        imagenes_a_procesar = state["imagenes_relevantes"][:max_imagenes_recuperadas]
        
        if len(state["imagenes_relevantes"]) > max_imagenes_recuperadas:
            print(f"   ⚠️ Limitando a {max_imagenes_recuperadas} imágenes (de {len(state['imagenes_relevantes'])}) por restricción de API.")

        # Añadir imágenes recuperadas al mensaje
        for i, img_path in enumerate(imagenes_a_procesar):
            try:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    user_content.append({
                        "type": "text", 
                        "text": f"\n[IMAGEN RECUPERADA {i+1}: {os.path.basename(img_path)}]"
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                    print(f"   🖼️ Adjuntando imagen al prompt: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"   ⚠️ Error cargando imagen {img_path}: {e}")

        # Si el usuario subió una imagen, también la adjuntamos
        if state.get('imagen_consulta') and os.path.exists(state['imagen_consulta']):
            try:
                with open(state['imagen_consulta'], "rb") as image_file:
                    query_image_data = base64.b64encode(image_file.read()).decode("utf-8")
                user_content.append({
                    "type": "text",
                    "text": "\n[IMAGEN DE CONSULTA DEL USUARIO]"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{query_image_data}"}
                })
            except Exception as e:
                print(f"   ⚠️ Error cargando imagen de consulta: {e}")

        messages.append(HumanMessage(content=user_content))


        response = await self.llm.ainvoke(messages)
        state["respuesta_final"] = response.content

        print("   ✅ Respuesta generada")

        state["trayectoria"].append({"nodo": "generar_respuesta", "timestamp": time.time()})
        cleanup_memory()
        return state

    async def _nodo_finalizar(self, state: AgentState) -> AgentState:
        # Guardar la interacción actual en la memoria a largo plazo
        # Limitar la respuesta final para el resumen para evitar saturar el LLM
        respuesta_final_val = state.get("respuesta_final", "")
        consulta_usuario_val = state.get("consulta_usuario", "")
        respuesta_corta = respuesta_final_val[:2000]
        
        # Generar un resumen rápido usando el LLM
        if self.memoria is not None and respuesta_final_val and consulta_usuario_val:
            try:
                print("   📝 Resumiendo interacción para la memoria...")
                messages = [
                    SystemMessage(content="Eres un asistente que resume interacciones. Escribe un resumen MUY BREVE (1-3 oraciones) de lo que preguntó el usuario y lo que respondiste o concluiste."),
                    HumanMessage(content=f"USUARIO: {consulta_usuario_val}\nASISTENTE: {respuesta_corta}")
                ]
                resume_response = await self.llm.ainvoke(messages)
                summary = resume_response.content
                
                self.memoria.add_interaction_summary(
                    session_id=state["user_id"],
                    user_query=consulta_usuario_val,
                    summary=f"Consulta: {consulta_usuario_val} | Resumen: {summary}"
                )
            except Exception as e:
                print(f"   ⚠️ Error generando resumen de interacción: {e}")
        state["trayectoria"].append({"nodo": "finalizar", "timestamp": time.time()})
        return state

    # ========== MÉTODOS DE PROCESAMIENTO ==========

    def _extraer_figuras_de_texto(self, texto: str) -> List[str]:
        """Extrae identificadores de figuras mencionadas en el texto de una página.
        Maneja ruido de OCR: middle dots (·), tildes (~), espacios, etc."""
        # Patrones con prefijo "Figura"
        patrones_figura = [
            r'[Ff]igura\s+(\d+[\-\.·]\d+)',
            r'[Ff]i[gG~][\.\s]*\s*(\d+[\-\.·\s]\d+)',
            r'FIGURA\s+(\d+[\-\.·]\d+)',
            r'[Ff]tg[\.\s]*\s*(\d+[\-\.·]\d+)',
        ]
        # Patrones con prefijo "Imagen"
        patrones_imagen = [
            r'[Ii]magen\s+(\d+[\-\.·]\d+)',
            r'IMAGEN\s+(\d+[\-\.·]\d+)',
        ]
        figuras = set()
        for patron in patrones_figura:
            matches = re.findall(patron, texto)
            for m in matches:
                normalizado = re.sub(r'[·\.\s]', '-', m)
                figuras.add(f"Figura {normalizado}")
        for patron in patrones_imagen:
            matches = re.findall(patron, texto)
            for m in matches:
                normalizado = re.sub(r'[·\.\s]', '-', m)
                figuras.add(f"Imagen {normalizado}")
        return sorted(list(figuras))

    def leer_pdf(self, archivo: str) -> List[Dict[str, Any]]:
        try:
            reader = PdfReader(archivo)
            paginas = []
            for i, page in enumerate(reader.pages):
                texto = page.extract_text()
                if texto:
                    paginas.append({"texto": texto, "pagina": i + 1})
            return paginas
        except Exception as e:
            print(f"❌ Error leyendo PDF: {e}")
            return []

    def split_texto(self, paginas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        size = Config.TEXT_CHUNK_SIZE
        overlap = Config.TEXT_CHUNK_OVERLAP
        
        for pagina in paginas:
            texto = pagina["texto"]
            num_pag = pagina["pagina"]
            if len(texto) < size:
                chunks.append({"texto": texto, "pagina": num_pag})
                continue
                
            for i in range(0, len(texto), size - overlap):
                chunk_text = texto[i:i + size]
                chunks.append({"texto": chunk_text, "pagina": num_pag})
                
        return chunks

    async def procesar_pdfs(self, archivos: List[str], forzar: bool = False):
        await self.gestor_qdrant.crear_colecciones()
        for archivo in archivos:
            if not os.path.exists(archivo): continue
            
            # Obtener texto por páginas
            paginas_info = self.leer_pdf(archivo)
            chunks_info = self.split_texto(paginas_info)
            
            # Reconstruir texto completo para ontología
            texto_completo = "\n".join([p["texto"] for p in paginas_info])
            
            imagenes = self.procesador.extraer_imagenes_pdf(archivo)
            if not self.ontologia:
                self.ontologia = self.extractor_ontologia.extraer_ontologia_completa(texto_completo, len(imagenes))
                
            await self._procesar_contenido_batch(chunks_info, None, archivo, tipo="texto")
            await self._procesar_contenido_batch(chunks_info, imagenes, archivo, tipo="imagen")
            cleanup_memory()

    async def _procesar_contenido_batch(self, chunks_info, imagenes, pdf_name, tipo="texto"):
        # items = chunks_info (List[Dict]) or imagenes (List[Dict])
        items = chunks_info if tipo == "texto" else imagenes
        batch_mv, batch_fde = [], []

        for i, item in enumerate(items):
            if tipo == "texto":
                contenido = item["texto"]
                page_num = item["pagina"]
                mv_embedding = self.procesador.generar_embedding_texto(contenido)
                payload = {
                    "pdf_name": str(pdf_name), 
                    "tipo": "texto", 
                    "texto": contenido[:500],
                    "numero_pagina": page_num,
                    "figuras": self._extraer_figuras_de_texto(contenido),
                    "nombre_archivo": Path(pdf_name).stem
                }
            else:
                # Imagen
                contenido = item["path"]
                page_num = item["page"]
                mv_embedding = self.procesador.generar_embedding_imagen(contenido)
                
                # Contexto de texto: Intentamos buscar chunks de la misma pagina
                contexto_texto = ""
                figura_asignada = []
                if chunks_info:
                    # Buscar chunks que coincidan con la pagina
                    chunks_pag = [c["texto"] for c in chunks_info if c["pagina"] == page_num]
                    if chunks_pag:
                        contexto_texto = " ".join(chunks_pag)
                    # Extraer todas las figuras de la página
                    figuras_en_pagina = self._extraer_figuras_de_texto(contexto_texto)
                    
                    # Asignación por posición: usar el índice de la imagen en la página
                    img_idx = item.get("img_index_in_page", 0)
                    total_imgs = item.get("total_images_in_page", 1)
                    
                    if figuras_en_pagina and total_imgs > 1:
                        # Ordenar figuras numéricamente para mapear con posición
                        figuras_ordenadas = sorted(figuras_en_pagina)
                        if img_idx < len(figuras_ordenadas):
                            # Asignar SOLO la figura correspondiente a esta posición
                            figura_asignada = [figuras_ordenadas[img_idx]]
                            print(f"      📍 Imagen #{img_idx+1}/{total_imgs} en Pg {page_num} → {figura_asignada[0]}")
                        else:
                            # Más imágenes que figuras: asignar todas como fallback
                            figura_asignada = figuras_en_pagina
                            print(f"      ⚠️ Pg {page_num}: Imagen #{img_idx+1} no tiene figura exacta (solo hay {len(figuras_ordenadas)} figs en texto)")
                    else:
                        # Solo 1 imagen en la página, o no hay figuras: asignar todas
                        figura_asignada = figuras_en_pagina

                payload = {
                    "pdf_name": str(pdf_name), 
                    "tipo": "imagen", 
                    "imagen_path": contenido, 
                    "contexto_texto": contexto_texto[:1000],
                    "numero_pagina": page_num,
                    "figuras": figura_asignada,
                    "nombre_archivo": Path(pdf_name).stem
                }

            if mv_embedding is None: continue
            fde_embedding = self.procesador.generar_fde_muvera(mv_embedding)

            # Generar un ID UUID determinístico basado en el contenido para evitar duplicados y errores 400
            seed_id = f"{Path(pdf_name).stem}_{tipo}_{page_num}_{i}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_id))

            batch_mv.append(PointStruct(id=point_id, vector=mv_embedding.tolist(), payload=payload))
            batch_fde.append(PointStruct(id=point_id, vector=fde_embedding.tolist(), payload=payload))

            if len(batch_mv) >= Config.BATCH_SIZE:
                await self.gestor_qdrant.insertar_batch_muvera(batch_mv, batch_fde)
                batch_mv, batch_fde = [], []
                cleanup_memory()

        if batch_mv:
            await self.gestor_qdrant.insertar_batch_muvera(batch_mv, batch_fde)

    async def procesar_consulta(self, consulta: str, imagen_path: Optional[str] = None, imagen_base64: Optional[str] = None, user_id: str = "default") -> str:
        initial_state = AgentState(
            messages=[], consulta_usuario=consulta, imagen_consulta=imagen_path,
            imagen_base64=imagen_base64,
            contexto_memoria="", ontologia=self.ontologia or {}, contexto_ontologico="",
            clasificacion="", consulta_optimizada="", filtros_ontologia=[],
            resultados_busqueda=[], contexto_documentos="", imagenes_relevantes=[],
            respuesta_final="", trayectoria=[], user_id=user_id, tiempo_inicio=time.time(),
            abortar_reset=False
        )
        config = {"configurable": {"thread_id": user_id}}
        final_state = await self.compiled_graph.ainvoke(initial_state, config=config)
        return final_state["respuesta_final"]

    def cerrar(self):
        cleanup_memory()

# ============================================================================
# COMPATIBILIDAD CON API EXISTENTE
# ============================================================================

class AsistenteHistologiaMultimodal(SistemaRAGColPaliPuro):
    """
    Clase de compatibilidad para mantener el backend API y tests funcionando
    """
    def __init__(self):
        super().__init__()
        self.collection_name = "histopatologia"

    @property
    def qdrant_client(self):
        return self.gestor_qdrant.client

    def _inicializar_modelos_embedding(self):
        """Compatibilidad con tests de embedding"""
        if self.procesador is None:
            self.procesador = ProcesadorColPaliPuro()

    def generate_image_embedding(self, image_path):
        """Compatibilidad con tests"""
        emb = self.procesador.generar_embedding_imagen(image_path)
        return emb.tolist() if emb is not None else None

    def generate_text_embedding(self, text):
        """Compatibilidad con tests"""
        emb = self.procesador.generar_embedding_texto(text)
        return emb.tolist() if emb is not None else None

    async def procesar_y_almacenar_pdfs_multimodal(self, pdf_files, use_muvera=True):
        """Alias para procesar_pdfs"""
        await self.procesar_pdfs([str(f) for f in pdf_files])

    async def iniciar_flujo_multimodal(self, consulta_usuario=None, imagen_path=None, imagen_base64=None, ground_truth=None):
        """Alias para procesar_consulta con formato de retorno anterior"""
        respuesta = await self.procesar_consulta(consulta_usuario or "Analizar contenido", imagen_path, imagen_base64)
        return {
            "respuesta": respuesta,
            "analisis_imagen": "Ver respuesta",
            "resultados_similares": await self.search_muvera(consulta_usuario, imagen_path),
            "scores_ragas": {},
            "tiempo": 0
        }

    async def search_muvera(self, query=None, image_path=None, top_k=5, prefetch_multiplier=5):
        """Compatibilidad con tests"""
        if image_path:
            query_mv = self.procesador.generar_embedding_imagen(image_path)
        else:
            query_mv = self.procesador.generar_embedding_texto(query or "")

        if query_mv is None:
            return {"pages": []}

        query_fde = self.procesador.generar_fde_muvera(query_mv)
        res, _ = await self.gestor_qdrant.buscar_muvera_2stage(query_mv, query_fde, top_k, prefetch_multiplier)
        return {"pages": res}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

_sistema_global = None

async def limpiar_colecciones(asistente):
    """Elimina las colecciones de Qdrant"""
    client = asistente.qdrant_client
    collections = [
        asistente.gestor_qdrant.content_mv_collection,
        asistente.gestor_qdrant.content_fde_collection
    ]

    print(f"\n🗑️ Limpiando {len(collections)} colecciones...")
    for col in collections:
        try:
            await client.delete_collection(col)
            print(f"   ✅ Eliminada: {col}")
        except Exception as e:
            print(f"   ⚠️ No se pudo eliminar {col}: {e}")

async def inicializar_sistema():
    global _sistema_global
    if _sistema_global is None:
        _sistema_global = AsistenteHistologiaMultimodal()
        _sistema_global.inicializar_componentes()
    return _sistema_global

async def main():
    try:
        sistema = await inicializar_sistema()
        print("\n🤖 Sistema RAG Multimodal listo (ColPali PURO)")
        while True:
            try:
                entrada = input(">> ").strip()
                if entrada.lower() in ['salir', 'exit', 'quit']: break
                if entrada:
                    res = await sistema.procesar_consulta(entrada)
                    print(f"\n📖 RESPUESTA:\n{res}\n")
            except KeyboardInterrupt: break
            except Exception as e: print(f"❌ Error: {e}")
    finally:
        if _sistema_global: _sistema_global.cerrar()

if __name__ == "__main__":
    asyncio.run(main())
