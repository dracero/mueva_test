"""
============================================================================
HISTOLOGÍA RAG MULTIMODAL - Local Script
============================================================================
Sistema RAG multimodal para histopatología con:
- ColPali/ColBERT para embeddings multimodales (late interaction)
- MUVERA para two-stage retrieval eficiente
- Gemini 2.5 Flash para generación de respuestas
- Ontología histopatológica normalizada (compatible SNOMED-CT/ICD-O3)
- RAGAS para evaluación de calidad
- LangSmith para telemetría

Autor: Consolidación de L5.ipynb + AsistenteHistologiaMultimodal
Uso: uv run muvera_test.py
============================================================================
"""
# ============================================================================
# IMPORTS
# ============================================================================
import os
import gc
import json
import time
import shutil
import asyncio
import base64
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfReader
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

# Async support
import nest_asyncio
nest_asyncio.apply()

# PyTorch
import torch

# Qdrant
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    MultiVectorConfig, MultiVectorComparator,
    Filter, HasIdCondition, HnswConfigDiff
)

# MUVERA from fastembed
# Corrected import for Muvera based on latest fastembed structure
from fastembed.postprocess import Muvera

# ColBERT - Late interaction text embeddings
from fastembed import LateInteractionTextEmbedding

# ColPali - Visual document embeddings
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import BitsAndBytesConfig

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Cargar credenciales desde variables de entorno
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# RAGAS
from ragas import evaluate, EvaluationDataset
try:
    from ragas.metrics.collections import (
        LLMContextRecall,
        Faithfulness,
        FactualCorrectness,
        ResponseRelevancy
    )
except ImportError:
    # Fallback for older ragas versions
    from ragas.metrics import (
        LLMContextRecall,
        Faithfulness,
        FactualCorrectness,
        ResponseRelevancy
    )
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

# Visualization
import plotly.io as pio
pio.renderers.default = "notebook"

# ============================================================================
# CONFIGURACIÓN DE CREDENCIALES
# ============================================================================
# Configurar gestión de memoria de PyTorch para evitar fragmentación
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Google API Key para Gemini
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""

# Qdrant Cloud credentials
COLLECTION_NAME = "documentos_pdf"

print("✅ Credenciales configuradas")
print(f"   QDRANT_URL: {QDRANT_URL[:30]}..." if QDRANT_URL else "   ⚠️ QDRANT_URL no configurada")
print(f"   Colección: {COLLECTION_NAME}")

# ============================================================================
# ONTOLOGÍA HISTOPATOLÓGICA NORMALIZADA
# ============================================================================
# Compatible con SNOMED-CT y clasificaciones ICD-O3 para histopatología

HISTOPATHOLOGY_ONTOLOGY = {
    # Tipos de tejidos principales
    "tissues": {
        "epithelial": {
            "simple": ["squamous", "cuboidal", "columnar"],
            "stratified": ["squamous", "cuboidal", "columnar", "transitional"],
            "pseudostratified": ["ciliated", "non-ciliated"],
            "glandular": ["exocrine", "endocrine", "mixed"]
        },
        "connective": {
            "proper": {
                "loose": ["areolar", "adipose", "reticular"],
                "dense": ["regular", "irregular", "elastic"]
            },
            "specialized": {
                "cartilage": ["hyaline", "elastic", "fibrocartilage"],
                "bone": ["compact", "spongy", "woven", "lamellar"],
                "blood": ["plasma", "formed_elements"],
                "lymphatic": ["lymph", "lymphoid_tissue"]
            }
        },
        "muscular": {
            "skeletal": ["type_I", "type_IIa", "type_IIb"],
            "cardiac": ["atrial", "ventricular", "purkinje"],
            "smooth": ["unitary", "multiunit"]
        },
        "nervous": {
            "neurons": ["multipolar", "bipolar", "unipolar", "pseudounipolar"],
            "glia": ["astrocytes", "oligodendrocytes", "microglia", "schwann_cells", "ependymal"]
        }
    },

    # Métodos de tinción histológica
    "staining": {
        "routine": {
            "H&E": {"hematoxylin": "basophilic", "eosin": "acidophilic"},
            "description": "Tinción de rutina para morfología general"
        },
        "special": {
            "PAS": {"target": "glycogen_mucopolysaccharides", "color": "magenta"},
            "Masson_Trichrome": {"target": "collagen", "colors": ["blue", "green"]},
            "Giemsa": {"target": "blood_parasites", "color": "purple_blue"},
            "Gomori": {"target": "reticular_fibers", "color": "black"},
            "Orcein": {"target": "elastic_fibers", "color": "brown"},
            "Sudan": {"target": "lipids", "color": "red_black"},
            "Silver_impregnation": {"target": "neurons_reticular", "color": "black_brown"}
        },
        "immunohistochemistry": {
            "markers": {
                "epithelial": ["cytokeratins", "EMA", "E-cadherin"],
                "mesenchymal": ["vimentin", "desmin", "SMA"],
                "neural": ["S100", "NSE", "synaptophysin", "chromogranin"],
                "lymphoid": ["CD3", "CD4", "CD8", "CD20", "CD45"],
                "proliferation": ["Ki-67", "PCNA"],
                "apoptosis": ["p53", "BCL2", "Caspase-3"]
            }
        }
    },

    # Estructuras celulares y extracelulares
    "structures": {
        "cellular": {
            "nucleus": ["chromatin", "nucleolus", "nuclear_envelope", "nuclear_pores"],
            "cytoplasm": ["organelles", "cytoskeleton", "inclusions"],
            "membrane": ["plasma_membrane", "glycocalyx", "cell_junctions"],
            "junctions": ["tight", "adherens", "desmosomes", "gap", "hemidesmosomes"]
        },
        "extracellular": {
            "matrix": {
                "fibrous": ["collagen", "elastin", "reticular"],
                "ground_substance": ["proteoglycans", "glycosaminoglycans", "glycoproteins"]
            },
            "basement_membrane": ["lamina_lucida", "lamina_densa", "lamina_reticularis"]
        }
    },

    # Patología histológica
    "pathology": {
        "neoplastic": {
            "benign": {
                "epithelial": ["adenoma", "papilloma", "polyp"],
                "mesenchymal": ["lipoma", "fibroma", "leiomyoma", "chondroma"]
            },
            "malignant": {
                "carcinoma": ["squamous", "adenocarcinoma", "transitional", "undifferentiated"],
                "sarcoma": ["fibrosarcoma", "liposarcoma", "leiomyosarcoma", "osteosarcoma"],
                "lymphoma": ["hodgkin", "non_hodgkin"],
                "other": ["melanoma", "glioma", "mesothelioma"]
            },
            "grading": ["well_differentiated", "moderately_differentiated", "poorly_differentiated"],
            "staging": ["TNM", "FIGO", "Dukes"]
        },
        "inflammatory": {
            "acute": ["neutrophilic", "serous", "fibrinous", "hemorrhagic"],
            "chronic": ["lymphocytic", "plasmacytic", "granulomatous", "eosinophilic"],
            "granulomatous": ["caseating", "non_caseating", "foreign_body", "immune"]
        },
        "degenerative": {
            "atrophy": ["physiological", "pathological", "disuse", "denervation"],
            "necrosis": ["coagulative", "liquefactive", "caseous", "fat", "fibrinoid", "gangrenous"],
            "apoptosis": ["intrinsic", "extrinsic"]
        },
        "adaptive": {
            "hypertrophy": ["compensatory", "hormonal", "workload"],
            "hyperplasia": ["physiological", "pathological"],
            "metaplasia": ["squamous", "intestinal", "osseous"],
            "dysplasia": ["mild", "moderate", "severe", "CIN", "PIN"]
        }
    },

    # Órganos y sistemas
    "organs": {
        "digestive": ["esophagus", "stomach", "small_intestine", "large_intestine", "liver", "pancreas", "gallbladder"],
        "respiratory": ["trachea", "bronchi", "bronchioles", "alveoli", "pleura"],
        "urinary": ["kidney", "ureter", "bladder", "urethra"],
        "reproductive": {
            "male": ["testis", "epididymis", "prostate", "seminal_vesicle"],
            "female": ["ovary", "uterus", "cervix", "vagina", "breast"]
        },
        "cardiovascular": ["heart", "arteries", "veins", "capillaries", "lymphatics"],
        "nervous": ["cerebrum", "cerebellum", "spinal_cord", "peripheral_nerves", "ganglia"],
        "endocrine": ["pituitary", "thyroid", "parathyroid", "adrenal", "pancreatic_islets"],
        "lymphoid": ["lymph_nodes", "spleen", "thymus", "tonsils", "MALT"],
        "integumentary": ["epidermis", "dermis", "hypodermis", "appendages"]
    }
}

def get_ontology_context(query: str) -> str:
    """
    Extrae contexto ontológico relevante basado en la consulta.
    Busca términos histopatológicos en la ontología para enriquecer la búsqueda.
    """
    query_lower = query.lower()
    context_parts = []

    # Buscar en tejidos
    for tissue_type, subtypes in HISTOPATHOLOGY_ONTOLOGY["tissues"].items():
        if tissue_type in query_lower:
            context_parts.append(f"Tejido {tissue_type}: {json.dumps(subtypes, ensure_ascii=False)[:200]}")

    # Buscar en tinciones
    for stain_cat, stains in HISTOPATHOLOGY_ONTOLOGY["staining"].items():
        for stain_name in (stains.keys() if isinstance(stains, dict) else []):
            if stain_name.lower().replace("_", " ") in query_lower or stain_name.lower() in query_lower:
                context_parts.append(f"Tinción {stain_name}: {json.dumps(stains.get(stain_name, {}), ensure_ascii=False)[:150]}")

    # Buscar en patología
    for path_type, subtypes in HISTOPATHOLOGY_ONTOLOGY["pathology"].items():
        if path_type in query_lower:
            context_parts.append(f"Patología {path_type}: {json.dumps(subtypes, ensure_ascii=False)[:200]}")

    # Buscar en órganos
    for system, organs in HISTOPATHOLOGY_ONTOLOGY["organs"].items():
        organ_list = organs if isinstance(organs, list) else []
        for organ in organ_list:
            if organ.replace("_", " ") in query_lower:
                context_parts.append(f"Órgano {organ} (Sistema {system})")

    return "\n".join(context_parts) if context_parts else ""

# ============================================================================
# CONFIGURACIÓN DE LANGSMITH (TELEMETRÍA)
# ============================================================================
def setup_langsmith_environment():
    """Configurar variables de entorno para LangSmith."""
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": LANGSMITH_API_KEY,
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "histologia_rag_multimodal_gemini"
    }

    for key, value in langsmith_config.items():
        if value:
            os.environ[key] = value
            print(f"✅ {key} configurado")

    try:
        from langsmith import traceable, Client
        client = Client()
        print(f"🔗 Conectado a LangSmith - Proyecto: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
        return True, traceable, client
    except Exception as e:
        print(f"⚠️ LangSmith no disponible: {e}")
        def dummy_traceable(*args, **kwargs):
            def decorator(func):
                return func
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator
        return False, dummy_traceable, None

LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith_environment()

# ============================================================================
# DECORADOR PARA MEDICIÓN DE ACCIONES
# ============================================================================
def medir_accion(nombre: str, tipo: str, extra_metadata: dict = None):
    """Decorador universal para medir todas las acciones del sistema."""
    def decorator(func):
        metadata = {
            "action_type": tipo,
            "function": func.__name__,
            "module": func.__module__
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        @traceable(name=nombre, run_type="chain", metadata=metadata)
        async def async_wrapper(*args, **kwargs):
            inicio = time.time()
            print(f"\n{'='*60}")
            print(f"🎯 ACCIÓN: {nombre} | TIPO: {tipo}")
            print(f"⏰ Inicio: {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            try:
                result = await func(*args, **kwargs)
                tiempo = time.time() - inicio
                print(f"✅ Éxito | ⏱️ Tiempo: {tiempo:.2f}s")
                return result
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                raise

        @traceable(name=nombre, run_type="chain", metadata=metadata)
        def sync_wrapper(*args, **kwargs):
            inicio = time.time()
            print(f"\n{'='*60}")
            print(f"🎯 ACCIÓN: {nombre} | TIPO: {tipo}")
            print(f"⏰ Inicio: {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            try:
                result = func(*args, **kwargs)
                tiempo = time.time() - inicio
                print(f"✅ Éxito | ⏱️ Tiempo: {tiempo:.2f}s")
                return result
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# ============================================================================
# DECORADOR PARA RETRY CON BACKOFF (RATE LIMITS)
# ============================================================================
def retry_with_backoff(max_retries: int = 3, base_delay: float = 5.0):
    """
    Decorador para reintentar llamadas a APIs con rate limiting.
    Usa exponential backoff para los reintentos.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"⏳ Rate limit alcanzado. Reintentando en {delay:.1f}s... (intento {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                        else:
                            print(f"❌ Rate limit: máximo de reintentos alcanzado")
                            raise
                    else:
                        raise
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"⏳ Rate limit alcanzado. Reintentando en {delay:.1f}s... (intento {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            print(f"❌ Rate limit: máximo de reintentos alcanzado")
                            raise
                    else:
                        raise
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

# ============================================================================
# CLASE PARA MÉTRICAS RAGAS
# ============================================================================
class MetricasRAGAS:
    """Evaluación de calidad RAG con RAGAS usando Gemini"""

    def __init__(self, google_api_key: str):
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0,
            max_output_tokens=2048,
        )
        self.evaluator_llm = LangchainLLMWrapper(gemini_llm)
        self.evaluaciones = []
        print("✅ Sistema de métricas RAGAS inicializado con Gemini")

    def preparar_datos_evaluacion(self, consulta: str, respuesta: str,
                                   contextos: List[str], ground_truth: Optional[str] = None) -> Dict:
        contextos_limpios = []
        for ctx in contextos:
            if ctx and isinstance(ctx, str) and ctx.strip():
                ctx_truncado = str(ctx).strip()[:2000]
                if ctx_truncado:
                    contextos_limpios.append(ctx_truncado)

        if not contextos_limpios:
            contextos_limpios = ["Sin contexto disponible"]

        datos = {
            "user_input": str(consulta).strip() if consulta else "Pregunta no especificada",
            "response": str(respuesta).strip() if respuesta else "Sin respuesta",
            "retrieved_contexts": contextos_limpios,
        }

        if ground_truth and str(ground_truth).strip():
            datos["reference"] = str(ground_truth).strip()

        return datos

    async def evaluar_respuesta(self, consulta: str, respuesta: str,
                                contextos: List[str], ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluar una respuesta usando métricas RAGAS"""
        print("\n📊 Evaluando con RAGAS (Gemini)...")

        try:
            datos = self.preparar_datos_evaluacion(consulta, respuesta, contextos, ground_truth)
            evaluation_dataset = EvaluationDataset.from_list([datos])

            metricas = [Faithfulness(), ResponseRelevancy()]
            if ground_truth:
                metricas.append(LLMContextRecall())
                metricas.append(FactualCorrectness())

            result = evaluate(dataset=evaluation_dataset, metrics=metricas, llm=self.evaluator_llm)

            scores = {}
            for metrica in metricas:
                try:
                    if hasattr(result, 'to_pandas'):
                        df = result.to_pandas()
                        if metrica.name in df.columns:
                            scores[metrica.name] = float(df[metrica.name].iloc[0])
                except Exception:
                    scores[metrica.name] = 0.0

            self.evaluaciones.append({
                "consulta": consulta[:200],
                "scores": scores,
                "timestamp": time.time()
            })

            print("📈 RESULTADOS RAGAS:")
            for metrica, valor in scores.items():
                emoji = "✅" if valor > 0.7 else "⚠️" if valor > 0.4 else "❌"
                print(f"  {emoji} {metrica}: {valor:.4f}")

            return scores

        except Exception as e:
            print(f"❌ Error en evaluación RAGAS: {e}")
            return {"faithfulness": 0.0, "factual_correctness": 0.0, "answer_relevancy": 0.0}

# ============================================================================
# CLASE PRINCIPAL - ASISTENTE DE HISTOLOGÍA MULTIMODAL
# ============================================================================
class AsistenteHistologiaMultimodal:
    """
    Sistema RAG multimodal para histopatología con:
    - ColPali/ColBERT embeddings
    - MUVERA two-stage retrieval
    - Gemini 2.5 Flash
    - Ontología histopatológica
    """

    def __init__(self):
        # Credenciales Qdrant
        self.qdrant_url = QDRANT_URL
        self.qdrant_api_key = QDRANT_KEY
        self.collection_name = "documentos_pdf"
        self.vector_store = None

        # Cliente Qdrant cacheado
        self._qdrant_client = None

        # Modelos
        self.llm = None
        self.colbert_model = None
        self.colpali_model = None
        self.colpali_processor = None

        # MUVERA configuration
        self.muvera = Muvera(
            dim=128,        # ColPali embedding dimensionality
            k_sim=6,        # 64 clusters (2^6)
            dim_proj=16,    # Compress to 16 dimensions per cluster
            r_reps=20,      # 20 repetitions
            random_seed=42,
        )

        # Dimensiones de embedding
        self.text_embedding_dim = 128   # ColPali uses 128 dim
        self.image_embedding_dim = 128  # ColPali uses 128 dim
        self.fde_dim = 1024             # MUVERA FDE dimension

        # Unified Collection (Pages)
        self.pages_collection = f"{self.collection_name}_pages_mv"
        self.pages_fde_collection = f"{self.collection_name}_pages_fde"
        
        # Legacy collections (kept for backward compatibility during migration if needed, but we will use pages)
        self.text_collection = f"{self.collection_name}_texto_mv"
        self.image_collection = f"{self.collection_name}_imagenes_mv"
        self.multimodal_collection = f"{self.collection_name}_multimodal_mv"

        # Colecciones FDE para MUVERA
        self.text_fde_collection = f"{self.collection_name}_texto_fde"
        self.image_fde_collection = f"{self.collection_name}_imagenes_fde"
        self.multimodal_fde_collection = f"{self.collection_name}_multimodal_fde"

        # Memoria y métricas
        self.memoria_semantica = None
        self.metricas_ragas = None
        self.temario = ""
        self.contenido_completo = ""
        self.trayectorias = []
        self.max_workers = 4

        print("✅ AsistenteHistologiaMultimodal inicializado")
        print(f"   📦 Colección base: {self.collection_name}")
        # muvera.k_sim determines clusters, number of clusters is usually implicit or defined in other ways in newer fastembed
        print(f"   🚀 MUVERA inicializado")

    @property
    def qdrant_client(self):
        """Cliente Qdrant cacheado"""
        if self._qdrant_client is None:
            self._qdrant_client = AsyncQdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=120,
                prefer_grpc=False,
                check_compatibility=False
            )
            print("🔗 Cliente Qdrant conectado")
        return self._qdrant_client

    @medir_accion("inicializar_componentes", "inicializacion")
    def inicializar_componentes(self):
        """Inicializar todos los componentes del asistente"""
        self._inicializar_llm()
        self._inicializar_modelos_embedding()
        self._inicializar_memoria()
        self._inicializar_ragas()

        # Verificar conexión Qdrant (auto-fix port)
        try:
            asyncio.create_task(self._verificar_conexion_qdrant())
        except:
            pass

        print("✅ Todos los componentes inicializados")

    async def _verificar_conexion_qdrant(self):
        """Verificar y corregir conexión a Qdrant si es necesario"""
        client = self.qdrant_client
        try:
            print("⏳ Verificando conexión Qdrant...")
            await client.get_collections()
            print("✅ Conexión Qdrant verificada")
        except Exception as e:
            print(f"⚠️ Error inicial conectando a Qdrant: {e}")
            # Intentar fix puerto 6333 si es error 404 o timeout y no tiene puerto
            if "404" in str(e) or "Not Found" in str(e) or "connect" in str(e).lower():
                current_url = self.qdrant_url
                if current_url and ":" not in current_url.split("/")[-1]:
                    new_url = f"{current_url}:6333"
                    print(f"🔄 Intentando reconectar con puerto 6333: {new_url}...")
                    self.qdrant_url = new_url
                    self._qdrant_client = AsyncQdrantClient(
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        timeout=120,
                        prefer_grpc=False,
                        check_compatibility=False
                    )
                    try:
                        await self._qdrant_client.get_collections()
                        print("✅ Conexión Qdrant recuperada con puerto :6333")
                        return
                    except Exception as e2:
                        print(f"❌ Falló re-conexión: {e2}")
            print("❌ No se pudo establecer conexión estable con Qdrant. Verifica QDRANT_URL.")

    def _inicializar_llm(self):
        """Inicializar Gemini 2.5 Flash"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_output_tokens=None,
        )
        print("✅ Gemini 2.5 Flash inicializado")

    def _inicializar_modelos_embedding(self):
        """Inicializar ColBERT y ColPali para embeddings multimodales"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📦 Cargando modelos de late interaction (device: {device})...")

        # ColBERT v2.0 para texto
        print("   📝 Inicializando ColBERT v2.0...")
        try:
            self.colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
            print(f"   ✅ ColBERT v2.0 cargado ({self.text_embedding_dim} dims)")
        except Exception as e:
            if "ONNXRuntimeError" in str(e) or "NoSuchFile" in str(e):
                print(f"   ⚠️ Error de caché detectado en ColBERT: {e}")
                print("   🔄 Intentando reparar caché...")
                import shutil
                cache_dir = Path("/tmp/fastembed_cache/models--colbert-ir--colbertv2.0")
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    print("   🗑️ Caché corrupta eliminada.")
                
                print("   ⬇️ Re-descargando modelo...")
                self.colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
                print(f"   ✅ ColBERT v2.0 recuperado y cargado")
            else:
                raise e

        # ColPali para imágenes
        # ColPali v1.2 for Visual Document Retrieval (4-bit Quantization)
        print("   🖼️ Inicializando ColPali v1.2 (4-bit)...")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            self.colpali_model = ColPali.from_pretrained(
                "vidore/colpali-v1.2",
                quantization_config=quantization_config,
                device_map="auto", # bitsandbytes manages device map
                torch_dtype=torch.bfloat16
            )
            self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            print(f"   ✅ ColPali v1.2 cargado (4-bit, {self.image_embedding_dim} dims)")
        except Exception as e:
            print(f"   ⚠️ Error cargando ColPali: {e}")
            if "bitsandbytes" in str(e):
                 print("   💡 Intenta: pip install bitsandbytes accelerate")
            self.colpali_model = None
            self.colpali_processor = None

    def _inicializar_memoria(self):
        """Inicializar memoria semántica"""
        self.memoria_semantica = self.SemanticMemory(llm=self.llm)
        print("✅ Memoria semántica inicializada")

    def _inicializar_ragas(self):
        """Inicializar sistema de métricas RAGAS"""
        try:
            if GOOGLE_API_KEY:
                self.metricas_ragas = MetricasRAGAS(GOOGLE_API_KEY)
            else:
                print("⚠️ GOOGLE_API_KEY no encontrada para RAGAS")
        except Exception as e:
            print(f"⚠️ Error inicializando RAGAS: {e}")
            self.metricas_ragas = None

    # ==================== MUVERA PROCESSING ====================

    def generate_muvera_fde(self, multivectors):
        """
        Generar Fixed Dimensional Encoding (FDE) usando MUVERA.
        Transforma multi-vector en single-vector para fast retrieval.
        Using process_document (correct method name).
        """
        mv = np.array(multivectors, dtype=np.float32)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)

        # Usar MUVERA para generar FDE
        # Updated method from process_doc to process_document
        fde = self.muvera.process_document(mv)
        return fde.tolist()

    def generate_muvera_query(self, query_multivectors):
        """Procesar query con MUVERA para búsqueda en colección FDE"""
        mv = np.array(query_multivectors, dtype=np.float32)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)
        return self.muvera.process_query(mv).tolist()

    # ==================== TEXT PROCESSING ====================

    @medir_accion("leer_pdf", "lectura", {"formato": "pdf"})
    def leer_pdf(self, nombre_archivo):
        """Leer contenido de texto de un PDF"""
        try:
            reader = PdfReader(nombre_archivo)
            texto = "".join(page.extract_text() or "" for page in reader.pages)
            print(f"📄 Leídos {len(texto)} caracteres de {nombre_archivo}")
            return texto
        except Exception as e:
            print(f"Error al leer {nombre_archivo}: {e}")
            return ""

    def split_into_chunks(self, text, chunk_size=2000):
        """Dividir texto en chunks"""
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"📄 Dividido en {len(chunks)} chunks")
        return chunks

    # @medir_accion("generate_text_embeddings", "procesamiento", {"modelo": "colbert"})
    def generate_text_embeddings(self, chunks, batch_size=32):
        """Generar embeddings ColBERT para texto"""
        embeddings = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = list(self.colbert_model.embed(batch))

            for emb in batch_embeddings:
                embeddings.append(emb.tolist())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress = (i + len(batch)) / total_chunks * 100
            print(f"🔄 Embedding ColBERT: {progress:.1f}%")

        print(f"✅ Generados {len(embeddings)} embeddings ColBERT")
        return embeddings

    # ==================== IMAGE PROCESSING ====================

    @medir_accion("extraer_imagenes_pdf", "procesamiento", {"formato": "pdf"})
    def extraer_imagenes_pdf(self, pdf_path, output_folder="extracted_images_histologia"):
        """Extraer imágenes de PDF"""
        os.makedirs(output_folder, exist_ok=True)
        imagenes = []

        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=300)

            for page_num, page in enumerate(pages):
                img_path = os.path.join(
                    output_folder,
                    f"{os.path.basename(pdf_path)}_page{page_num}.png"
                )
                page.save(img_path, 'PNG')
                imagenes.append(img_path)

            print(f"✅ Extraídas {len(imagenes)} páginas de {pdf_path}")
            return imagenes

        except Exception as e:
            print(f"⚠️ Error extrayendo imágenes: {e}")
            return []

    # ==================== PAGE-LEVEL EXTRACTION HELPERS ====================

    def extract_text_and_images_per_page(self, pdf_path):
        """
        Extrae texto e imagen página por página para mantener la relación multimodal.
        Retorna una lista de diccionarios: [{'page': 1, 'text': '...', 'image_path': '...'}]
        """
        results = []
        try:
            # 1. Extract Text per page
            reader = PdfReader(pdf_path)
            
            # 2. Extract Images per page (using pdf2image)
            from pdf2image import convert_from_path
            # Convert request: dpi=200 is enough for embedding, saves RAM
            try:
                images = convert_from_path(pdf_path, dpi=200)
            except Exception as e:
                print(f"⚠️ Error convirtiendo PDF a imágenes: {e}")
                images = []

            # Guardar imágenes temporalmente
            temp_dir = "extracted_images_histologia"
            os.makedirs(temp_dir, exist_ok=True)
            
            total_pages = len(reader.pages)
            
            for i in range(total_pages):
                page_data = {"page": i + 1, "text": "", "image_path": None}
                
                # Texto
                try:
                    page_data["text"] = reader.pages[i].extract_text() or ""
                except Exception:
                    pass
                
                # Imagen
                if i < len(images):
                    image_name = f"{os.path.basename(pdf_path)}_page_{i+1}.png"
                    image_path_full = os.path.join(temp_dir, image_name)
                    images[i].save(image_path_full, "PNG")
                    page_data["image_path"] = image_path_full
                
                results.append(page_data)
                
            return results
            
        except Exception as e:
            print(f"❌ Error procesando páginas de {pdf_path}: {e}")
            return []

    # @medir_accion("generate_image_embedding", "procesamiento", {"modelo": "colpali"})
    def generate_image_embedding(self, image_path):
        """Generar embedding ColPali para imagen histológica"""
        try:
            if self.colpali_model is None:
                return None

            image = Image.open(image_path).convert("RGB")
            batch_images = self.colpali_processor.process_images([image])
            batch_images = {k: v.to(self.colpali_model.device) for k, v in batch_images.items()}

            with torch.no_grad():
                image_embeddings = self.colpali_model(**batch_images)

            multivector = image_embeddings[0].cpu().float().numpy().tolist()
            return multivector

        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None

    # @medir_accion("generate_image_embeddings_batch", "procesamiento", {"modelo": "colpali"})
    def generate_image_embeddings_batch(self, image_paths, batch_size=4):
        """Procesar múltiples imágenes en batch"""
        if self.colpali_model is None:
            return [None] * len(image_paths)

        embeddings = []
        total = len(image_paths)

        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]

            try:
                images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                    except Exception:
                        embeddings.append(None)

                if not images:
                    continue

                batch_inputs = self.colpali_processor.process_images(images)
                batch_inputs = {k: v.to(self.colpali_model.device) for k, v in batch_inputs.items()}

                with torch.no_grad():
                    batch_embeddings = self.colpali_model(**batch_inputs)

                for emb in batch_embeddings:
                    embeddings.append(emb.cpu().float().numpy().tolist())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error en batch: {e}")
                embeddings.extend([None] * len(batch_paths))

        return embeddings

    def generate_multimodal_embedding(self, text, image_path=None):
        """Generar embedding multimodal combinando texto e imagen"""
        multivectors = []

        if text:
            text_embeddings = list(self.colbert_model.embed([text]))[0]
            multivectors.extend(text_embeddings.tolist())

        if image_path:
            image_embedding = self.generate_image_embedding(image_path)
            if image_embedding is not None:
                multivectors.extend(image_embedding)
    # ==================== QDRANT STORAGE ====================

    # @medir_accion("store_in_qdrant", "escritura_db", {"db": "qdrant"})
    async def store_in_qdrant(self, points, collection_name, embedding_dim=128, is_multivector=True):
        """Almacenar puntos en Qdrant"""
        client = self.qdrant_client

        # Ensure collection exists (create if not found)
        try:
            print(f"   🔍 Verificando colección '{collection_name}'...")
            await client.get_collection(collection_name)
            print(f"   📂 Colección '{collection_name}' ya existe")
        except Exception as e:
            # Si error es "Not Found", creamos. Si es otro, lo mostramos.
            status_code = getattr(e, "status_code", None)
            if status_code != 404 and "Not Found" not in str(e):
                print(f"   ⚠️ Error verificando colección {collection_name}: {e}")
            
            print(f"   ✨ Creando colección '{collection_name}' (Multivector: {is_multivector})...")
            try:
                if is_multivector:
                    # Disable HNSW for MV collections - saves RAM, brute-force MaxSim is used anyway
                    await client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=embedding_dim,
                            distance=Distance.COSINE,
                            multivector_config=MultiVectorConfig(
                                comparator=MultiVectorComparator.MAX_SIM
                            )
                        ),
                        hnsw_config=HnswConfigDiff(m=0)
                    )
                else:
                    await client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
                print(f"   ✅ Colección '{collection_name}' creada exitosamente")
            except Exception as create_err:
                print(f"   ❌ Error CRÍTICO creando colección {collection_name}: {create_err}")
                import traceback
                traceback.print_exc()
                raise create_err

        # Insert points (OUTSIDE the create try-except, always runs)
        try:
            # Validación de diagnóstico antes de insertar
            if points and len(points) > 0:
                sample_vec = points[0].vector
                print(f"   🧐 Diagnóstico Vector: Tipo={type(sample_vec)}, Len={len(sample_vec) if hasattr(sample_vec, '__len__') else 'N/A'}")
                if is_multivector and isinstance(sample_vec, list) and len(sample_vec) > 0:
                    print(f"   🧐 Muestra MV[0]: Tipo={type(sample_vec[0])}, Len={len(sample_vec[0]) if hasattr(sample_vec[0], '__len__') else 'N/A'}")

            # Insertar en chunks para evitar timeouts/payload limits
            CHUNK_SIZE = 5 if is_multivector else 50
            total_points = len(points)
            
            for i in range(0, total_points, CHUNK_SIZE):
                chunk = points[i : i + CHUNK_SIZE]
                print(f"   💾 Insertando chunk {i//CHUNK_SIZE + 1}/{(total_points-1)//CHUNK_SIZE + 1} ({len(chunk)} puntos) en '{collection_name}'...")
                await client.upsert(collection_name=collection_name, points=chunk, wait=True)
                
            print(f"   ✅ Almacenados {total_points} elementos en '{collection_name}'")
        except Exception as e:
            print(f"   ❌ Error insertando puntos en {collection_name}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    # @medir_accion("store_with_muvera", "escritura_db", {"tipo": "muvera_dual"})
    async def store_with_muvera(self, mv_points, collection_base, embedding_dim=128):
        """Almacenar con estructura dual MUVERA (FDE + MV)"""
        fde_points = []

        print(f"🔄 Generando FDEs MUVERA para {len(mv_points)} puntos...")

        for p in mv_points:
            fde = self.generate_muvera_fde(p.vector)
            fde_points.append(PointStruct(
                id=p.id,
                vector=fde,
                payload={**p.payload, "mv_id": p.id}
            ))

        # Colección multi-vector
        mv_collection = f"{collection_base}_mv"
        await self.store_in_qdrant(mv_points, mv_collection, embedding_dim, is_multivector=True)

        # Colección FDE
        fde_collection = f"{collection_base}_fde"
        fde_dim = len(fde_points[0].vector) if fde_points else self.fde_dim
        await self.store_in_qdrant(fde_points, fde_collection, fde_dim, is_multivector=False)
    @medir_accion("procesar_pdfs_multimodal", "procesamiento", {"pipeline": "pure_colpali"})
    async def procesar_y_almacenar_pdfs_multimodal(self, pdf_files, use_muvera=True):
        """Procesar PDFs página por página con ColPali (Visual Document Retrieval)"""
        page_points = []
        global_id_counter = int(time.time())

        print(f"\n📚 Procesando {len(pdf_files)} PDFs con Pure ColPali")
        inicio_total = time.time()

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
                continue

            print(f"\n📄 Procesando: {pdf_file}")
            
            # Extract text per page for metadata
            reader_for_text = PdfReader(str(pdf_file))
            
            # Extract images for each page (render page as image)
            # convert_from_path renders the full page
            try:
                from pdf2image import convert_from_path
                pages_images = convert_from_path(str(pdf_file), dpi=150) # 150 dpi is usually enough for embeddings
                
                # Ensure output directory exists
                os.makedirs("extracted_images_histologia", exist_ok=True)
                
                print(f"   🖼️ Renderizadas {len(pages_images)} páginas.")
                
                for page_num, page_img in enumerate(pages_images, start=1):
                    # Save temp image for processing
                    temp_img_path = f"temp_page_{page_num}.png"
                    page_img.save(temp_img_path, "PNG")
                    
                    try:
                        # Generate ColPali Embedding for the FULL PAGE
                        # ColPali accepts images and generates multivectors
                        print(f"      🔹 Embdedding Pág {page_num}...")
                        
                        # Process single image
                        batch_images = self.colpali_processor.process_images([page_img])
                        batch_images = {k: v.to(self.colpali_model.device) for k, v in batch_images.items()}

                        with torch.no_grad():
                            image_embeddings = self.colpali_model(**batch_images)

                        multivector = image_embeddings[0].cpu().float().numpy().tolist()
                        
                        # Extract text for this page (for metadata)
                        page_text_preview = ""
                        if page_num - 1 < len(reader_for_text.pages):
                            try:
                                page_text_preview = (reader_for_text.pages[page_num - 1].extract_text() or "")[:500]
                            except Exception:
                                pass
                        
                        # Create payload
                        page_points.append(PointStruct(
                            id=global_id_counter,
                            vector=multivector,
                            payload={
                                "pdf_name": str(pdf_file),
                                "page_number": page_num,
                                "type": "page",
                                "image_path": temp_img_path,
                                "text_preview": page_text_preview,
                                "domain": "histologia"
                            }
                        ))
                        global_id_counter += 1
                        
                        # Cleanup VRAM aggressively
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"❌ Error procesando página {page_num}: {e}")
                    finally:
                        # Move temp image to permanent folder
                        final_path = os.path.join("extracted_images_histologia", f"{os.path.basename(pdf_file)}_page_{page_num}.png")
                        if os.path.exists(temp_img_path):
                            shutil.move(temp_img_path, final_path)
                        # update/correct payload path in last point
                        if page_points:
                            page_points[-1].payload["image_path"] = os.path.basename(final_path)

            except Exception as e:
                print(f"❌ Error leyendo PDF como imágenes: {e}")

        # Almacenar en Qdrant (Unified Pages Collection)
        print("\n💾 Almacenando Páginas en Qdrant...")
        
        if page_points:
             # Using store_with_muvera to create both _mv and _fde collections
             await self.store_with_muvera(page_points, f"{self.collection_name}_pages", self.image_embedding_dim)

        tiempo_total = time.time() - inicio_total
        print(f"\n✅ Procesamiento completado en {tiempo_total:.1f}s")
        print(f"   - {len(page_points)} páginas indexadas")

    # ==================== BÚSQUEDA MULTIMODAL ====================

    @medir_accion("search_muvera", "busqueda", {"tipo": "pure_colpali"})
    async def search_muvera(self, query=None, image_path=None, top_k=5, prefetch_multiplier=5):
        """
        Búsqueda 2-stage con MUVERA sobre Colección de PÁGINAS.
        Unified retrieval for text query or image query.
        """
        client = self.qdrant_client
        results = {"pages": []}

        print(f"\n🚀 BÚSQUEDA MUVERA (Pure ColPali)")
        
        collection_base = f"{self.collection_name}_pages"
        fde_collection = f"{collection_base}_fde"
        mv_collection = f"{collection_base}_mv"
        
        try:
            query_mv_list = []
            
            # Generate Query Embedding (Text or Image) using ColPali
            if image_path:
                print(f"   🖼️ Query: Imagen ({os.path.basename(image_path)})")
                query_mv_list = self.generate_image_embedding(image_path)
            elif query:
                print(f"   📝 Query: Texto ('{query[:50]}...')")
                # ColPali Text Query Embedding
                # processor.process_queries handles text
                batch_queries = self.colpali_processor.process_queries([query])
                batch_queries = {k: v.to(self.colpali_model.device) for k, v in batch_queries.items()}
                with torch.no_grad():
                    query_embeddings = self.colpali_model(**batch_queries)
                query_mv_list = query_embeddings[0].cpu().float().numpy().tolist()
            else:
                return results

            if not query_mv_list:
                print("❌ No se pudo generar embedding para la query")
                return results

            # Generate FDE
            query_fde = self.generate_muvera_fde(query_mv_list) # FDE logic is generic for multivectors

            # STAGE 1: Fast FDE retrieval
            print(f"   📂 Stage 1 - FDE search: {fde_collection}")
            try:
                fde_response = await client.query_points(
                    collection_name=fde_collection,
                    query=query_fde,
                    limit=top_k * prefetch_multiplier
                )
                if not fde_response.points:
                    print(f"      ⚠️ No se encontraron candidatos en Stage 1")
                    return results
                    
                candidate_ids = [point.id for point in fde_response.points]
                print(f"      ✅ Stage 1: {len(candidate_ids)} candidatos")
                
                # STAGE 2: MV Reranking
                print(f"   📂 Stage 2 - MV reranking: {mv_collection}")
                reranked_results = await client.query_points(
                    collection_name=mv_collection,
                    query=query_mv_list,
                    query_filter=Filter(
                        must=[HasIdCondition(has_id=candidate_ids)]
                    ),
                    limit=top_k,
                )
                
                results["pages"] = [{
                    "id": r.id,
                    "score": round(r.score, 4),
                    "payload": r.payload
                } for r in reranked_results.points]
                print(f"      ✅ Stage 2: {len(results['pages'])} páginas recuperadas")
                
            except Exception as e:
                print(f"   ⚠️ Error en búsqueda: {e}")
                if "404" not in str(e):
                     import traceback
                     traceback.print_exc()

            return results

        except Exception as e:
            print(f"❌ Error general en búsqueda: {e}")
            return results

    # ==================== ANÁLISIS DE IMÁGENES ====================

    @retry_with_backoff(max_retries=3, base_delay=8.0)
    @medir_accion("analizar_imagen_histologica", "agente", {"tipo": "vision_histologia"})
    async def analizar_imagen_histologica(self, image_path):
        """Analizar imagen histológica con Gemini Vision y ontología"""
        try:
            print(f"\n🔬 ANALIZANDO IMAGEN: {os.path.basename(image_path)}")

            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode()

            # Buscar imágenes similares con mayor recall
            resultados_similares = await self.search_muvera(image_path=image_path, top_k=5)

            contexto_similar = "\n".join([
                f"Página similar (Score: {r['score']}): PDF {r['payload'].get('pdf_name', 'N/A')} - Pág {r['payload'].get('page_number', '?')}"
                for r in resultados_similares.get('pages', [])
            ])

            # Prompt con ontología
            ontology_summary = json.dumps({
                "tissues": list(HISTOPATHOLOGY_ONTOLOGY["tissues"].keys()),
                "staining": list(HISTOPATHOLOGY_ONTOLOGY["staining"].keys()),
                "pathology": list(HISTOPATHOLOGY_ONTOLOGY["pathology"].keys())
            }, ensure_ascii=False)

            prompt = f"""Analiza esta imagen histológica usando la siguiente ontología histopatológica:
{ontology_summary}

Proporciona:
1. TIPO_TEJIDO: Clasificación según ontología (epitelial, conectivo, muscular, nervioso)
2. ESTRUCTURAS: Estructuras celulares y extracelulares visibles
3. COLORACION: Tipo de tinción (H&E, PAS, Masson, etc.)
4. CARACTERISTICAS: Morfología distintiva
5. ORGANO_PROBABLE: Identificación del órgano
6. PATOLOGIA: Si hay alteraciones (neoplásicas, inflamatorias, degenerativas)
7. DESCRIPCION_DETALLADA: Análisis histológico completo

Contexto de imágenes similares:
{contexto_similar}"""

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"}
                ]
            )

            respuesta = await self.llm.ainvoke([message])
            print(f"✅ Análisis completado")
            return respuesta.content

        except Exception as e:
            print(f"❌ Error analizando imagen: {e}")
            return None

    # ==================== VERIFICACIÓN ESTUDIANTE ====================

    @medir_accion("verificar_estudiante", "evaluacion", {"tipo": "docencia"})
    async def verificar_descripcion_estudiante(self, image_path: str, descripcion_estudiante: str):
        """
        Verifica la descripción de un estudiante comparándola con el análisis experto generado.
        """
        print(f"\n🎓 VERIFICANDO DESCRIPCIÓN DE ESTUDIANTE")

        # 1. Generar análisis experto (Ground Truth)
        analisis_experto = await self.analizar_imagen_histologica(image_path)
        if not analisis_experto:
            return "❌ No se pudo generar el análisis experto para verificación."

        # 2. Comparar usando LLM
        prompt_evaluacion = f"""Actúa como un profesor estricto pero constructivo de histopatología.

        Tienes dos descripciones de la misma imagen microscópica:
        1. ANÁLISIS EXPERTO (Ground Truth):
        {analisis_experto}

        2. DESCRIPCIÓN DEL ESTUDIANTE:
        "{descripcion_estudiante}"

        Tu tarea es evaluar el desempeño del estudiante.
        Genera un reporte con:
        - **Puntaje estimado (0-10)** basado en precisión terminológica y diagnóstica.
        - **Aciertos clave**: Qué identificó correctamente el estudiante.
        - **Errores u omisiones**: Qué confundió o qué detalles importantes le faltaron.
        - **Recomendación de estudio**: Qué temas de la ontología debería repasar.

        Sé directo y pedagógico."""

        try:
            message = HumanMessage(content=prompt_evaluacion)
            respuesta_evaluacion = await self.llm.ainvoke([message])
            return respuesta_evaluacion.content
        except Exception as e:
            print(f"❌ Error en evaluación de estudiante: {e}")
            return "Error generando la evaluación."

    # ==================== GENERACIÓN DE RESPUESTAS ====================

    @retry_with_backoff(max_retries=3, base_delay=8.0)
    @medir_accion("generate_answer", "generacion", {"modelo": "gemini"})
    async def generate_answer(self, query_text: str, contexts: List[str],
                              image_paths: List[str] = None) -> str:
        """
        Generar respuesta usando Gemini Vision.
        """
        # Enriquecer con ontología
        ontology_context = get_ontology_context(query_text)

        system_prompt = """Eres un profesor experto en Histología e Histopatología.
Responde usando la información de los documentos proporcionados.
Usa terminología histológica precisa y describe las estructuras con detalle.
Si hay imágenes, analízalas y relácionaalas con la teoría.
Responde en Markdown y destaca los conceptos importantes."""

        user_content = [{"type": "text", "text": f"""Consulta: {query_text}

Contexto ontológico histopatológico:
{ontology_context}

Documentos recuperados:
{chr(10).join(contexts[:5])}"""}]

        # Añadir imágenes si las hay
        if image_paths:
            for img_path in image_paths[:5]:
                try:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    user_content.append({
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_data}"
                    })
                except Exception:
                    pass

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    # ==================== FLUJO PRINCIPAL ====================

    @medir_accion("flujo_multimodal", "pipeline", {"sistema": "completo"})
    async def iniciar_flujo_multimodal(self, consulta_usuario: str = None,
                                        imagen_path: str = None,
                                        ground_truth: Optional[str] = None):
        """Flujo completo de RAG multimodal con evaluación RAGAS"""
        print(f"\n{'='*80}")
        print(f"🔬 FLUJO RAG MULTIMODAL HISTOLOGÍA")
        print(f"📝 Consulta: {consulta_usuario[:80] if consulta_usuario else 'Análisis de imagen'}")
        print(f"{'='*80}")

        inicio = time.time()

        try:
            # Análisis de imagen si existe
            analisis_imagen = None
            if imagen_path:
                analisis_imagen = await self.analizar_imagen_histologica(imagen_path)

            # Búsqueda MUVERA 2-stage (Aumentar top_k para mejor recall)
            resultados = await self.search_muvera(
                query=consulta_usuario,
                image_path=imagen_path,
                top_k=10
            )

            # Preparar contextos (En Pure ColPali, los contextos son las imágenes de las páginas + metadatos)
            # Para Gemini Flash, pasaremos las imágenes de las páginas recuperadas como input visual
            contextos_texto = []
            retrieved_page_images = []
            
            for r in resultados.get('pages', []):
                # Metadata textual enriched with text preview and page number
                text_preview = r['payload'].get('text_preview', '')
                page_ref = f"Referencia: PDF {r['payload'].get('pdf_name')} - Pág {r['payload'].get('page_number')} (Score: {r.get('score', 'N/A')})"
                if text_preview:
                    page_ref += f"\nContenido de la página: {text_preview[:300]}"
                contextos_texto.append(page_ref)
                
                # Imagen de la página recuperada
                img_name = r['payload'].get('image_path')
                if img_name:
                    full_path = os.path.join("extracted_images_histologia", img_name)
                    if os.path.exists(full_path):
                        retrieved_page_images.append(full_path)

            print(f"   📄 Contextos recuperados: {len(contextos_texto)}")
            print(f"   🖼️ Páginas visuales para contexto: {len(retrieved_page_images)}")

            # Combinar imágenes: La imagen subida por usuario (si hay) + Páginas recuperadas
            # Limitamos a top 3 páginas recuperadas para no saturar tokens/quota
            final_image_inputs = []
            if imagen_path:
                final_image_inputs.append(imagen_path)
            
            final_image_inputs.extend(retrieved_page_images[:3])

            respuesta = await self.generate_answer(
                query_text=consulta_usuario or "Explicar la imagen histológica",
                contexts=contextos_texto,
                image_paths=final_image_inputs 
            )

            # Evaluación RAGAS (Opcional, skip si hay error de cuota)
            scores_ragas = {}
            if self.metricas_ragas and respuesta and contextos_texto:
                print("\n📊 Evaluando con RAGAS (Gemini)...")
                try:
                    scores_ragas = await self.metricas_ragas.evaluar_respuesta(
                        consulta=consulta_usuario or "Análisis de imagen",
                        respuesta=respuesta,
                        contextos=contextos_texto,
                        ground_truth=None # No tenemos GT por ahora
                    )
                    print(f"   📈 Faithfulness: {scores_ragas.get('faithfulness', 'N/A')}")
                    print(f"   🎯 Relevancy: {scores_ragas.get('answer_relevancy', 'N/A')}")
                except Exception as e:
                    print(f"   ⚠️ Saltando evaluación RAGAS por error: {e}")
                    # Si es Rate Limit, no pasa nada, ya tenemos la respuesta

            # Actualizar memoria
            if self.memoria_semantica:
                self.memoria_semantica.add_interaction(
                    consulta_usuario or "Análisis de imagen",
                    respuesta or ""
                )

            tiempo_total = time.time() - inicio

            print(f"\n{'='*80}")
            print("🔬 RESPUESTA:")
            print("="*80)
            print(respuesta or "No se generó respuesta")
            print(f"\n⏱️ Tiempo total: {tiempo_total:.2f}s")
            print("="*80)

            return {
                "respuesta": respuesta,
                "analisis_imagen": analisis_imagen,
                "resultados_similares": resultados,
                "scores_ragas": scores_ragas,
                "tiempo": tiempo_total
            }

        except Exception as e:
            print(f"❌ Error en flujo: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ==================== MEMORIA SEMÁNTICA ====================

    class SemanticMemory:
        def __init__(self, llm, max_entries=10):
            self.llm = llm
            self.conversations = []
            self.max_entries = max_entries
            self.summary = ""
            self.message_history = ChatMessageHistory()

        def add_interaction(self, query, response):
            self.message_history.add_user_message(query)
            self.message_history.add_ai_message(response)
            self.conversations.append({"query": query, "response": response})

            if len(self.conversations) > self.max_entries:
                self.conversations.pop(0)

        def get_context(self):
            return "\n".join([
                f"Q: {c['query'][:100]}\nA: {c['response'][:100]}"
                for c in self.conversations[-5:]
            ])

        def clear(self):
            self.conversations = []
            self.message_history.clear()

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def pil_image_to_base64(image: Image.Image) -> str:
    """Convertir imagen PIL a base64"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

async def limpiar_colecciones(asistente):
    """Eliminar colecciones de histología"""
    client = asistente.qdrant_client
    base = asistente.collection_name

    collections = [
        f"{base}_texto_mv", f"{base}_texto_fde",
        f"{base}_imagenes_mv", f"{base}_imagenes_fde",
        f"{base}_multimodal_mv", f"{base}_multimodal_fde",
    ]

    for collection in collections:
        try:
            await client.delete_collection(collection)
            print(f"✅ Colección '{collection}' eliminada")
        except Exception:
            pass

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

async def main():
    """Función principal de ejemplo"""
    print("\n" + "="*80)
    print("🔬 HISTOLOGÍA RAG MULTIMODAL - Local Execution")
    print("   🧬 ColBERT + ColPali | 🚀 MUVERA | 🤖 Gemini 2.5 Flash")
    print("="*80)

    # Inicializar asistente
    asistente = AsistenteHistologiaMultimodal()
    asistente.inicializar_componentes()

    # Opción de limpiar colecciones
    limpiar = input("¿Deseas limpiar las colecciones anteriores? (s/n): ").strip().lower()
    if limpiar == 's':
        print("\n🗑️ Limpiando colecciones anteriores...")
        await limpiar_colecciones(asistente)

    # Buscar PDFs en el directorio local ./pdfs/
    pdf_dir = Path("./pdfs")
    if pdf_dir.exists():
        archivos_existentes = list(pdf_dir.glob("*.pdf"))
    else:
        # Buscar en el directorio actual
        archivos_existentes = list(Path(".").glob("*.pdf"))

    print(f"\n📊 PDFs encontrados: {len(archivos_existentes)}")

    if archivos_existentes:
        procesar = input("¿Deseas procesar los PDFs encontrados (esto puede tardar)? (s/n): ").strip().lower()
        if procesar == 's':
            print("\n🔄 Procesando PDFs con MUVERA...")
            await asistente.procesar_y_almacenar_pdfs_multimodal(
                archivos_existentes,
                use_muvera=True
            )
    else:
        print("⚠️ No se encontraron PDFs. Coloca tus archivos PDF en ./pdfs/ o en el directorio actual.")

    print("\n✅ Sistema listo.")

    while True:
        print("\n" + "-"*50)
        mode = input("Selecciona modo: \n1. Consulta RAG Normal\n2. Verificar Descripción de Estudiante (Imagen + Texto)\n3. Salir\nOpción: ")

        if mode == "3":
            break

        elif mode == "1":
            consulta = input("Introduce tu consulta: ")
            await asistente.iniciar_flujo_multimodal(consulta_usuario=consulta)

        elif mode == "2":
            img_path = input("Ruta de la imagen a evaluar: ").strip()
            # Eliminar comillas si el usuario las pone
            img_path = img_path.replace('"', '').replace("'", "")

            if not os.path.exists(img_path):
                print(f"❌ Error: La imagen {img_path} no existe.")
                continue

            descripcion = input("Introduce la descripción del estudiante de la imagen:\n")

            # Ejecutar verificación
            feedback = await asistente.verificar_descripcion_estudiante(img_path, descripcion)

            print("\n📝 RESULTADO DE LA EVALUACIÓN:")
            print(feedback)

    return asistente

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # En Google Colab ejecutar: asistente = await main()
    # En Python estándar:
    try:
        asistente = asyncio.run(main())
    except KeyboardInterrupt:
        print("Saliendo...")