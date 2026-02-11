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
- Gemini 2.5 Flash: Generación de respuestas

VENTAJAS vs versión con ColBERT:
✅ Más simple (1 modelo en lugar de 2)
✅ Menos memoria GPU (~30% reducción)
✅ Consistencia total (mismo espacio de embeddings)
✅ ColPali maneja texto nativamente
✅ Código ~20% más corto
"""

import os
import json
import time
import asyncio
import base64
import uuid
import nest_asyncio
from typing import TypedDict, Annotated, List, Dict, Any, Optional
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
from PIL import Image, ImageEnhance
import numpy as np

# PyTorch
import torch

# Qdrant
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    MultiVectorConfig, MultiVectorComparator,
    Filter, HasIdCondition
)

# MUVERA from fastembed
from fastembed.postprocess import Muvera

# ColPali - Visual document embeddings
from colpali_engine.models import ColPali as ColPaliModel
from colpali_engine.models import ColPaliProcessor

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Gemini para extracción de ontología
import google.generativeai as genai

# Configuración de credenciales local
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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
    FDE_DIM = 1024               # MUVERA FDE dimension

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
        os.environ["LANGCHAIN_PROJECT"] = "rag_histopatologia_colpali_puro"
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
    """Extrae ontología histopatológica usando Gemini"""

    def __init__(self, api_key: str):
        if not api_key:
            print("⚠️ API Key de Google no proporcionada para ExtractorOntologia")
            self.model = None
            return
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

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

Formato JSON estructurado y compacto."""

        try:
            response = self.model.generate_content(prompt)
            ontologia_texto = response.text.replace("```json", "").replace("```", "").strip()
            ontologia = json.loads(ontologia_texto)

            ontologia["metadata"] = {
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "modelo": "gemini-2.5-flash",
                "num_imagenes": num_imagenes
            }

            with open(Config.ONTOLOGY_FILE, 'w', encoding='utf-8') as f:
                json.dump(ontologia, f, indent=2, ensure_ascii=False)

            print(f"✅ Ontología extraída: {len(ontologia)} categorías")
            return ontologia

        except Exception as e:
            print(f"⚠️ Error ontología: {e}")
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
            self.colpali_model = ColPaliModel.from_pretrained(
                "vidore/colpali-v1.2",
                torch_dtype=torch.bfloat16,
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
        """Extrae páginas como imágenes del PDF"""
        print(f"📄 Extrayendo páginas de {pdf_path}...")

        imagenes = []
        nombre_base = Path(pdf_path).stem
        
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
            # cleanup_memory() - Removido para optimizar velocidad en procesos por lote

            return multivector

        except Exception as e:
            print(f"❌ Error generando embedding imagen: {e}")
            return None

    def generar_embedding_imagen_batch(self, imagenes_paths: List[str]) -> List[np.ndarray]:
        """
        Genera embeddings ColPali multi-vector para un batch de imágenes
        """
        if self.colpali_model is None or not imagenes_paths:
            return []

        try:
            images = [self._preprocesar_imagen(path) for path in imagenes_paths]

            batch_images = self.colpali_processor.process_images(images)
            batch_images = {k: v.to(self.colpali_model.device) for k, v in batch_images.items()}

            with torch.no_grad():
                image_embeddings = self.colpali_model(**batch_images)

            # Extraer multivectores para cada imagen en el batch
            multivectors = [image_embeddings[i].cpu().float().numpy() for i in range(len(images))]

            del images, batch_images, image_embeddings
            return multivectors

        except Exception as e:
            print(f"❌ Error generando embeddings imagen batch: {e}")
            return []

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
            # cleanup_memory() - Removido para optimizar velocidad en procesos por lote
            
            return multivector

        except Exception as e:
            print(f"❌ Error generando embedding texto: {e}")
            return None

    def generar_embedding_texto_batch(self, textos: List[str]) -> List[np.ndarray]:
        """
        Genera embeddings ColPali multi-vector para un batch de textos
        """
        if self.colpali_model is None or not textos:
            return []

        try:
            batch_queries = self.colpali_processor.process_queries(textos)
            batch_queries = {k: v.to(self.colpali_model.device) for k, v in batch_queries.items()}

            with torch.no_grad():
                text_embeddings = self.colpali_model(**batch_queries)

            # Extraer multivectores para cada texto en el batch
            multivectors = [text_embeddings[i].cpu().float().numpy() for i in range(len(textos))]

            del batch_queries, text_embeddings
            return multivectors

        except Exception as e:
            print(f"❌ Error generando embeddings texto batch: {e}")
            return []

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
        prefetch_multiplier: int = 5
    ) -> List[Dict]:
        """
        Búsqueda 2-stage con MUVERA
        """
        client = self.client

        try:
            # STAGE 1: Fast FDE search
            fde_response = await client.query_points(
                collection_name=self.content_fde_collection,
                query=query_fde.tolist(),
                limit=top_k * prefetch_multiplier
            )
            
            if not fde_response.points:
                return []
            
            candidate_ids = [point.id for point in fde_response.points]

            # STAGE 2: Precise multi-vector reranking
            mv_response = await client.query_points(
                collection_name=self.content_mv_collection,
                query=query_multivector.tolist(),
                query_filter=Filter(
                    must=[HasIdCondition(has_id=candidate_ids)]
                ),
                limit=top_k
            )

            return [{
                "id": r.id,
                "score": float(r.score),
                "payload": r.payload
            } for r in mv_response.points]

        except Exception as e:
            print(f"❌ Error búsqueda MUVERA: {e}")
            return []

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
    user_id: str
    tiempo_inicio: float

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
        self.memory_saver = MemorySaver()

    def _setup_apis(self):
        """Configurar APIs"""
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        setup_langsmith()

    def inicializar_componentes(self):
        """Inicializar todos los componentes"""
        print("\n" + "="*80)
        print("🚀 SISTEMA RAG HISTOPATOLOGÍA - ColPali PURO + MUVERA + LangGraph")
        print("="*80)

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        # Procesador ColPali PURO
        self.procesador = ProcesadorColPaliPuro()

        # Qdrant
        self.gestor_qdrant = GestorQdrantMuvera(
            url=QDRANT_URL or "http://localhost:6333",
            api_key=QDRANT_KEY,
            collection_base="histopatologia"
        )

        # Extractor de ontología
        self.extractor_ontologia = ExtractorOntologia(GOOGLE_API_KEY)
        self.ontologia = self.extractor_ontologia.cargar_ontologia()

        # LangGraph
        self._inicializar_langgraph()
        cleanup_memory()

    def _inicializar_langgraph(self):
        """Inicializar grafo de agentes"""
        graph = StateGraph(AgentState)

        graph.add_node("inicializar", self._nodo_inicializar)
        graph.add_node("analizar_ontologia", self._nodo_analizar_ontologia)
        graph.add_node("clasificar", self._nodo_clasificar)
        graph.add_node("optimizar_consulta", self._nodo_optimizar_consulta)
        graph.add_node("buscar", self._nodo_buscar)
        graph.add_node("generar_respuesta", self._nodo_generar_respuesta)
        graph.add_node("finalizar", self._nodo_finalizar)

        graph.add_edge(START, "inicializar")
        graph.add_edge("inicializar", "analizar_ontologia")
        graph.add_edge("analizar_ontologia", "clasificar")
        graph.add_edge("clasificar", "optimizar_consulta")
        graph.add_edge("optimizar_consulta", "buscar")
        graph.add_edge("buscar", "generar_respuesta")
        graph.add_edge("generar_respuesta", "finalizar")
        graph.add_edge("finalizar", END)

        self.compiled_graph = graph.compile(checkpointer=self.memory_saver)

    # ========== NODOS DEL GRAFO ==========

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        state["ontologia"] = self.ontologia or {}
        state["tiempo_inicio"] = time.time()
        state["trayectoria"] = [{"nodo": "inicializar", "timestamp": time.time()}]
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
        if state.get('imagen_consulta') and os.path.exists(state['imagen_consulta']):
            query_mv = self.procesador.generar_embedding_imagen(state['imagen_consulta'])
        else:
            query_mv = self.procesador.generar_embedding_texto(state['consulta_optimizada'])

        if query_mv is not None:
            query_fde = self.procesador.generar_fde_muvera(query_mv)
            resultados = await self.gestor_qdrant.buscar_muvera_2stage(query_mv, query_fde)

        state["resultados_busqueda"] = resultados
        contextos = []
        imagenes = []
        for r in resultados:
            tipo = r['payload'].get('tipo', 'unknown')
            if tipo == 'texto':
                contextos.append(f"[TEXTO]\n{r['payload'].get('texto', '')[:400]}")
            elif tipo == 'imagen':
                img_path = r['payload'].get('imagen_path')
                if img_path:
                    imagenes.append(img_path)
                    contextos.append(f"[IMAGEN]\nImagen: {os.path.basename(img_path)}\nContexto: {r['payload'].get('contexto_texto', '')[:200]}")

        state["contexto_documentos"] = "\n\n".join(contextos)
        state["imagenes_relevantes"] = imagenes
        state["trayectoria"].append({"nodo": "buscar", "timestamp": time.time()})
        return state

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        """Nodo 6: Generar respuesta multimodal"""
        print("\n💭 Generando respuesta multimodal...")

        contexto_imagen = ""
        if state.get('imagen_consulta'):
            contexto_imagen = f"\n\nIMAGEN DE CONSULTA: {state['imagen_consulta']}\nEl usuario proporcionó una imagen. Analízala en contexto."

        if state["imagenes_relevantes"]:
            contexto_imagen += f"\n\nIMÁGENES RELEVANTES: {len(state['imagenes_relevantes'])} encontradas"

        messages = [
            SystemMessage(content="""Eres un profesor experto en histopatología con capacidad multimodal.

INSTRUCCIONES:
- Proporciona respuestas detalladas y didácticas
- Integra información textual e visual
- Usa terminología histológica precisa
- Referencias imágenes cuando sean relevantes
- Explica conceptos para estudiantes de medicina"""),
            HumanMessage(content=f"""CONSULTA: {state["consulta_usuario"]}
{contexto_imagen}

CLASIFICACIÓN: {state["clasificacion"]}

CONTEXTO ONTOLÓGICO:
{state["contexto_ontologico"]}

DOCUMENTOS Y CONTEXTOS RECUPERADOS:
{state["contexto_documentos"][:6000]}...

Genera una respuesta completa integrando toda la información disponible.""")
        ]

        response = await self.llm.ainvoke(messages)
        state["respuesta_final"] = response.content

        print("   ✅ Respuesta generada")

        state["trayectoria"].append({"nodo": "generar_respuesta", "timestamp": time.time()})
        cleanup_memory()
        return state

    async def _nodo_finalizar(self, state: AgentState) -> AgentState:
        state["trayectoria"].append({"nodo": "finalizar", "timestamp": time.time()})
        return state

    # ========== MÉTODOS DE PROCESAMIENTO ==========

    def leer_pdf(self, archivo: str) -> str:
        try:
            reader = PdfReader(archivo)
            texto = "".join(page.extract_text() or "" for page in reader.pages)
            return texto
        except Exception as e:
            print(f"❌ Error leyendo PDF: {e}")
            return ""

    def split_texto(self, texto: str) -> List[str]:
        size = Config.TEXT_CHUNK_SIZE
        overlap = Config.TEXT_CHUNK_OVERLAP
        return [texto[i:i + size] for i in range(0, len(texto), size - overlap)]

    async def procesar_pdfs(self, archivos: List[str], forzar: bool = False):
        await self.gestor_qdrant.crear_colecciones()
        for archivo in archivos:
            if not os.path.exists(archivo): continue
            texto = self.leer_pdf(archivo)
            chunks = self.split_texto(texto)
            imagenes = self.procesador.extraer_imagenes_pdf(archivo)
            if not self.ontologia:
                self.ontologia = self.extractor_ontologia.extraer_ontologia_completa(texto, len(imagenes))
            await self._procesar_contenido_batch(chunks, None, archivo, tipo="texto")
            await self._procesar_contenido_batch(chunks, imagenes, archivo, tipo="imagen")
            cleanup_memory()

    async def _procesar_contenido_batch(self, chunks, imagenes, pdf_name, tipo="texto"):
        contenidos = chunks if tipo == "texto" else [img["path"] for img in imagenes]

        # Procesar en batches de tamaño Config.BATCH_SIZE para optimizar GPU y reducir overhead
        batch_size = Config.BATCH_SIZE

        for i in range(0, len(contenidos), batch_size):
            batch_items = contenidos[i : i + batch_size]
            batch_mv, batch_fde = [], []

            # Generar embeddings en batch (MUCHO más rápido que uno a uno)
            if tipo == "texto":
                mv_embeddings = self.procesador.generar_embedding_texto_batch(batch_items)
            else:
                mv_embeddings = self.procesador.generar_embedding_imagen_batch(batch_items)

            if not mv_embeddings:
                continue

            for j, mv_embedding in enumerate(mv_embeddings):
                idx = i + j
                contenido_item = batch_items[j]

                if tipo == "texto":
                    payload = {"pdf_name": str(pdf_name), "tipo": "texto", "texto": contenido_item[:500]}
                else:
                    payload = {
                        "pdf_name": str(pdf_name),
                        "tipo": "imagen",
                        "imagen_path": contenido_item,
                        "contexto_texto": chunks[idx] if idx < len(chunks) else ""
                    }

                fde_embedding = self.procesador.generar_fde_muvera(mv_embedding)

                # Generar un ID UUID determinístico basado en el contenido para evitar duplicados
                seed_id = f"{Path(pdf_name).stem}_{tipo}_{idx}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_id))

                batch_mv.append(PointStruct(id=point_id, vector=mv_embedding.tolist(), payload=payload))
                batch_fde.append(PointStruct(id=point_id, vector=fde_embedding.tolist(), payload=payload))

            if batch_mv:
                await self.gestor_qdrant.insertar_batch_muvera(batch_mv, batch_fde)
                # Cleanup después de procesar y subir un batch completo
                cleanup_memory()

    async def procesar_consulta(self, consulta: str, imagen_path: Optional[str] = None, user_id: str = "default") -> str:
        initial_state = AgentState(
            messages=[], consulta_usuario=consulta, imagen_consulta=imagen_path,
            contexto_memoria="", ontologia=self.ontologia or {}, contexto_ontologico="",
            clasificacion="", consulta_optimizada="", filtros_ontologia=[],
            resultados_busqueda=[], contexto_documentos="", imagenes_relevantes=[],
            respuesta_final="", trayectoria=[], user_id=user_id, tiempo_inicio=time.time()
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

    async def iniciar_flujo_multimodal(self, consulta_usuario=None, imagen_path=None, ground_truth=None):
        """Alias para procesar_consulta con formato de retorno anterior"""
        respuesta = await self.procesar_consulta(consulta_usuario or "Analizar contenido", imagen_path)
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
        res = await self.gestor_qdrant.buscar_muvera_2stage(query_mv, query_fde, top_k, prefetch_multiplier)
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
