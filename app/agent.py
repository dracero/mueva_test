
from typing import AsyncIterable
import base64


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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Google GenAI para extracción de ontología
from google import genai as GoogleGenAIClient

# API Key Rotator (proyecto raíz)
import sys as _sys
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)
from api_key_rotator import google_key_rotator, create_google_llm, _is_quota_error  # noqa: E402

# Configuración de credenciales local
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

nest_asyncio.apply()
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*token_type_ids is not provided.*")

def redimensionar_y_codificar_imagen(img_path: str, max_dim: int = 384) -> str:
    """Abre una imagen, la redimensiona manteniendo el aspect ratio, y la devuelve en base64."""
    try:
        with Image.open(img_path) as img:
            # Convertir a RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar si supera max_dim
            w, h = img.size
            if max(w, h) > max_dim:
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            # Guardar en buffer como JPEG comprimido
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Error redimensionando imagen {img_path}: {e}")
        # Fallback al archivo crudo en base64 si falla PIL
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración del sistema ColPali Puro + MUVERA"""

    # Adaptado para ejecución local
    BASE_DIR = Path(__file__).resolve().parent.parent / "histopatologia_data"
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
    SEARCH_PREFETCH_MULTIPLIER = 20

    # --- ESTRATEGIA DE UMBRAL (configurable por GPU desde .env) ---
    #
    # El score absoluto de ColPali MaxSim varía por arquitectura GPU:
    #   GTX 1070 (Pascal FP32): correcto ~878, falso ~846  → threshold 868 funciona
    #   RTX 3050 (Ampere 4-bit): correcto ~804, falso ~?   → threshold debe calibrarse
    #
    # SEARCH_SCORE_THRESHOLD: umbral absoluto. Leído desde .env (SEARCH_SCORE_THRESHOLD).
    #   - Si se define en .env, ese valor se usa (permite calibrar por GPU).
    #   - Si es 0.0 (default): se usa top-k puro sin filtro absoluto.
    #
    # Con normalización L2 activa (NORMALIZE_EMBEDDINGS=True), después de re-indexar,
    # los scores estarán en un rango más compacto y comparable entre GPUs.
    SEARCH_SCORE_THRESHOLD = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.0"))
    NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

    # Cuantización: 8 = mejor precisión en scores (~870+), 4 = menos VRAM (~800 scores)
    _default_bits = "8"
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb < 7.0:
                _default_bits = "4"
                print(f"ℹ️ GPU VRAM detectada ({vram_gb:.2f} GB) es menor a 7 GB. Forzando default QUANTIZATION_BITS=4 para evitar CUDA OOM.")
    except Exception:
        pass
    QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", _default_bits))

    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.EMBEDDINGS_DIR, cls.ONTOLOGY_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

def normalizar_ruta_imagen(ruta: str) -> str:
    if not ruta:
        return ""
    nombre_base = os.path.basename(ruta)
    return str(Config.EMBEDDINGS_DIR / nombre_base)

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
    try:
        # Evitar importaciones durante el apagado (ImportError) usando variables globales directas
        if 'torch' in globals() and torch is not None:
            if hasattr(torch, "cuda") and torch.cuda is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
    except BaseException:
        pass

    try:
        if 'gc' in globals() and gc is not None:
            gc.collect()
    except BaseException:
        pass

# ============================================================================
# EXTRACTOR DE ONTOLOGÍA
# ============================================================================

class ExtractorOntologia:
    """Extrae ontología histopatológica usando Google Gemini"""

    def __init__(self, api_key: str = None, rotator=None):
        self._rotator = rotator or google_key_rotator
        resolved_key = api_key or self._rotator.get_key()
        if not resolved_key:
            print("⚠️ API Key de Google no proporcionada para ExtractorOntologia")
            self.model = None
            return
        self._current_key = resolved_key
        self._gemini_client = GoogleGenAIClient.Client(api_key=resolved_key)
        self.model = "gemini-2.5-flash"

    @staticmethod
    def extraer_caption_imagen(page_fitz, img_bbox, texto_pagina_completo: str) -> str:
        """Extrae la etiqueta 'Imagen X.X' / 'Fig X.X' + TODO el texto debajo de la imagen."""
        import re
        import fitz
        caption = ""
        try:
            page_rect = page_fitz.rect
            margen_overlap = 10
            area_expandida = fitz.Rect(0, max(0, img_bbox[3] - margen_overlap), page_rect.width, page_rect.height)
            texto_expandido = page_fitz.get_text("text", clip=area_expandida).strip()
            
            if texto_expandido:
                caption = texto_expandido
            else:
                area_abajo = fitz.Rect(0, img_bbox[3], page_rect.width, page_rect.height)
                caption = page_fitz.get_text("text", clip=area_abajo).strip()
        except Exception:
            pass
        
        if caption:
            caption = re.sub(r'\n\s*\d{1,3}\s*$', '', caption).strip()
            return caption
        return texto_pagina_completo[:500] if texto_pagina_completo else ""

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
                
                # Retry loop for rate limits (429/403) con rotación de key
                for attempt in range(5):
                    try:
                        response = self._gemini_client.models.generate_content(
                            model=self.model,
                            contents=prompt_actual,
                        )
                        break
                    except Exception as e:
                        err_str = str(e)
                        if _is_quota_error(e) and attempt < 4:
                            # Rotar key
                            self._rotator.report_failure(self._current_key)
                            new_key = self._rotator.get_key()
                            self._current_key = new_key
                            self._gemini_client = GoogleGenAIClient.Client(api_key=new_key)
                            print(f"⚠️ [Gemini Rate Limit - Ontología] Key rotada → ...{new_key[-4:]}. Reintentando inmediatamente... (Intento {attempt+1}/5)")
                            time.sleep(0.1)
                        else:
                            raise e
                            
                ontologia_texto = response.text.strip()
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
                    "modelo": self.model,
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
        bits = Config.QUANTIZATION_BITS
        print(f"   📚 Cargando ColPali v1.2 ({bits}-bit, texto + imágenes)...")
        try:
            from transformers import BitsAndBytesConfig

            # Limpiar VRAM antes de cargar para maximizar espacio disponible
            cleanup_memory()

            # Forzar kernels genéricos para máxima compatibilidad con GPUs nuevas (Blackwell sm_120)
            if torch.cuda.is_available():
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cudnn.enabled = False

            quantization_config = None
            if bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            
            kwargs = {
                "device_map": self.device if self.device == "cuda" else "cpu",
                "low_cpu_mem_usage": True,
            }
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            else:
                kwargs["torch_dtype"] = torch.bfloat16

            self.colpali_model = ColPaliModel.from_pretrained(
                "vidore/colpali-v1.2",
                **kwargs
            )
            self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            self.colpali_model.eval()
            print(f"   ✅ ColPali cargado ({bits}-bit, {Config.COLPALI_EMBEDDING_DIM}D multi-vector en {self.device})")
        except Exception as e:
            if bits == 8 or self.device == "cuda":
                print(f"   ⚠️ Error cargando ColPali en GPU, intentando fallback... ({e})")
                try:
                    # Liberar la carga parcial
                    if hasattr(self, 'colpali_model') and self.colpali_model is not None:
                        del self.colpali_model
                        self.colpali_model = None
                    cleanup_memory()

                    # Si falló 8-bit pero tenemos GPU, intentamos 4-bit en GPU
                    if bits == 8 and self.device == "cuda":
                        print("   🔄 Intentando cargar en 4-bit en GPU...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4"
                        )
                        self.colpali_model = ColPaliModel.from_pretrained(
                            "vidore/colpali-v1.2",
                            quantization_config=quantization_config,
                            device_map="cuda",
                            low_cpu_mem_usage=True,
                        )
                        self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
                        self.colpali_model.eval()
                        print(f"   ✅ ColPali cargado (4-bit fallback en GPU, {Config.COLPALI_EMBEDDING_DIM}D multi-vector)")
                    else:
                        # Si falló 4-bit o no tenemos GPU, intentamos en CPU bfloat16
                        raise ValueError("Fallback a CPU necesario")
                except Exception as e2:
                    print(f"   ⚠️ Error en fallback de GPU, intentando fallback a CPU (bfloat16): {e2}")
                    try:
                        if hasattr(self, 'colpali_model') and self.colpali_model is not None:
                            del self.colpali_model
                            self.colpali_model = None
                        cleanup_memory()
                        self.colpali_model = ColPaliModel.from_pretrained(
                            "vidore/colpali-v1.2",
                            torch_dtype=torch.bfloat16,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                        )
                        self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
                        self.colpali_model.eval()
                        self.device = "cpu"
                        print(f"   ✅ ColPali cargado (CPU bfloat16 fallback, {Config.COLPALI_EMBEDDING_DIM}D multi-vector)")
                    except Exception as e3:
                        print(f"   ❌ Error crítico cargando ColPali en CPU: {e3}")
                        self.colpali_model = None
                        self.colpali_processor = None
            else:
                print(f"   ❌ Error cargando ColPali: {e}")
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
                    
                    # Obtener el texto completo de la página para el fallback del caption
                    texto_pagina_completo = page.get_text("text").strip()
                    
                    # MÉTODO PRIMARIO: get_image_info(xrefs=True) devuelve SOLO las imágenes
                    # físicamente dibujadas en esta página (no el diccionario global del PDF)
                    img_info_list = page.get_image_info(xrefs=True)
                    page_xrefs = [info["xref"] for info in img_info_list if info.get("xref")]
                    
                    # Guardamos la posición Y y el bbox completo
                    page_y_positions = {info["xref"]: info.get("bbox", (0,0,0,0))[1] for info in img_info_list if info.get("xref")}
                    page_bboxes = {info["xref"]: info.get("bbox", (0,0,0,0)) for info in img_info_list if info.get("xref")}
                    
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
                                bbox = page_bboxes.get(xref, (0,0,0,0))
                                
                                # Extraer caption usando la nueva función
                                caption_extraido = ExtractorOntologia.extraer_caption_imagen(page, bbox, texto_pagina_completo)
                                
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
                                    
                                    # Magnificar imágenes pequeñas a mínimo 868px
                                    # para que ColPali genere embeddings de alta calidad
                                    target_size = 868
                                    if max(width, height) < target_size:
                                        scale = target_size / max(width, height)
                                        new_width = int(width * scale)
                                        new_height = int(height * scale)
                                        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                        width, height = img_pil.size
                                        area = width * height
                                    
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
                                        "y_position": y_position,
                                        "caption": caption_extraido
                                    })
                                else:
                                    os.remove(img_path)
                            except Exception as e:
                                print(f"⚠️ Error procesando xref {xref} en página {page_num+1}: {e}")
                    else:
                        print(f"      ⚠️ Pg {page_num+1}: Imagen incrustada como vector o Form XObject. Omitiendo captura de pantalla para evitar imágenes de texto.")
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
            
        # Fallback global deshabilitado para evitar imágenes de texto
        # Si PyMuPDF no puede extraer figuras, no queremos screenshots llenos de texto.
        if len(imagenes) == 0:
            print("⚠️ No se pudieron extraer figuras puras del PDF. Se omiten imágenes de texto descriptivo.")
        
        return imagenes

    def _preprocesar_imagen(self, imagen_path: str) -> Image.Image:
        """Preprocesamiento específico para histopatología"""
        image = Image.open(imagen_path).convert("RGB")

        # Ajustar tamaño de la imagen para que su lado más largo sea exactamente 868px.
        # Esto optimiza el consumo de VRAM y evita CUDA OOM en imágenes muy grandes (fotos, capturas de pantalla, etc.)
        width, height = image.size
        target_size = 868
        if max(width, height) != target_size:
            scale = target_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if Config.ENHANCE_CONTRAST:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(Config.CONTRAST_FACTOR)

        if Config.ENHANCE_BRIGHTNESS:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(Config.BRIGHTNESS_FACTOR)

        return image

    def generar_embedding_imagen(self, imagen_path: str) -> Optional[np.ndarray]:
        """
        Genera embedding ColPali multi-vector para imagen.
        Si falla por CUDA OOM, reintenta moviendo temporalmente a CPU.
        """
        if self.colpali_model is None:
            print("⚠️ ColPali no disponible")
            return None

        cleanup_memory()
        try:
            image = self._preprocesar_imagen(imagen_path)
            
            batch_images = self.colpali_processor.process_images([image])
            device = self.colpali_model.device
            batch_images = {k: v.to(device) for k, v in batch_images.items()}

            with torch.no_grad():
                if "cuda" in str(device):
                    with torch.cuda.amp.autocast():
                        image_embeddings = self.colpali_model(**batch_images)
                else:
                    image_embeddings = self.colpali_model(**batch_images)

            # Multi-vector output (convertir tensor a numpy)
            multivector = image_embeddings[0].cpu().float().numpy()

            # Normalización L2 por vector
            if Config.NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(multivector, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                multivector = multivector / norms

            print(f"   📊 [Debug] Embedding imagen: shape={multivector.shape} "
                  f"| norm_media={np.linalg.norm(multivector, axis=-1).mean():.4f} "
                  f"| normalizado={Config.NORMALIZE_EMBEDDINGS}")

            del image, batch_images, image_embeddings
            cleanup_memory()
            return multivector

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"⚠️ CUDA OOM al generar embedding imagen, reintentando en CPU...")
                cleanup_memory()
                try:
                    # Mover inputs a CPU y ejecutar en CPU
                    image = self._preprocesar_imagen(imagen_path)
                    batch_images = self.colpali_processor.process_images([image])
                    batch_images = {k: v.to("cpu") for k, v in batch_images.items()}
                    
                    # Temporalmente mover modelo a CPU
                    original_device = self.colpali_model.device
                    self.colpali_model = self.colpali_model.to("cpu")
                    
                    with torch.no_grad():
                        image_embeddings = self.colpali_model(**batch_images)
                    
                    multivector = image_embeddings[0].cpu().float().numpy()
                    
                    if Config.NORMALIZE_EMBEDDINGS:
                        norms = np.linalg.norm(multivector, axis=-1, keepdims=True)
                        norms = np.where(norms < 1e-8, 1.0, norms)
                        multivector = multivector / norms
                    
                    print(f"   ✅ Embedding generado en CPU (fallback): shape={multivector.shape}")
                    
                    del image, batch_images, image_embeddings
                    
                    # Intentar mover modelo de vuelta a GPU
                    try:
                        cleanup_memory()
                        self.colpali_model = self.colpali_model.to(original_device)
                    except Exception:
                        print("   ⚠️ No se pudo devolver modelo a GPU, se queda en CPU")
                        self.device = "cpu"
                    
                    return multivector
                except Exception as cpu_err:
                    print(f"❌ Error generando embedding imagen en CPU fallback: {cpu_err}")
                    cleanup_memory()
                    return None
            else:
                print(f"❌ Error generando embedding imagen: {e}")
                cleanup_memory()
                return None
        except Exception as e:
            print(f"❌ Error generando embedding imagen: {e}")
            cleanup_memory()
            return None

    def generar_embedding_texto(self, texto: str) -> Optional[np.ndarray]:
        """
        Genera embedding ColPali multi-vector para TEXTO.
        Si falla por CUDA OOM, reintenta en CPU.
        """
        if self.colpali_model is None:
            print("⚠️ ColPali no disponible")
            return None

        cleanup_memory()
        try:
            # ColPali procesa queries textuales
            batch_queries = self.colpali_processor.process_queries([texto])
            device = self.colpali_model.device
            batch_queries = {k: v.to(device) for k, v in batch_queries.items()}
            
            with torch.no_grad():
                if "cuda" in str(device):
                    with torch.cuda.amp.autocast():
                        text_embeddings = self.colpali_model(**batch_queries)
                else:
                    text_embeddings = self.colpali_model(**batch_queries)
            
            # Multi-vector output
            multivector = text_embeddings[0].cpu().float().numpy()

            # Normalización L2 (mismo tratamiento que embeddings de imagen)
            if Config.NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(multivector, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                multivector = multivector / norms

            del batch_queries, text_embeddings
            cleanup_memory()
            return multivector

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"⚠️ CUDA OOM al generar embedding texto, reintentando en CPU...")
                cleanup_memory()
                try:
                    batch_queries = self.colpali_processor.process_queries([texto])
                    batch_queries = {k: v.to("cpu") for k, v in batch_queries.items()}
                    
                    original_device = self.colpali_model.device
                    self.colpali_model = self.colpali_model.to("cpu")
                    
                    with torch.no_grad():
                        text_embeddings = self.colpali_model(**batch_queries)
                    
                    multivector = text_embeddings[0].cpu().float().numpy()
                    
                    if Config.NORMALIZE_EMBEDDINGS:
                        norms = np.linalg.norm(multivector, axis=-1, keepdims=True)
                        norms = np.where(norms < 1e-8, 1.0, norms)
                        multivector = multivector / norms
                    
                    print(f"   ✅ Embedding texto generado en CPU (fallback)")
                    
                    del batch_queries, text_embeddings
                    
                    try:
                        cleanup_memory()
                        self.colpali_model = self.colpali_model.to(original_device)
                    except Exception:
                        print("   ⚠️ No se pudo devolver modelo a GPU, se queda en CPU")
                        self.device = "cpu"
                    
                    return multivector
                except Exception as cpu_err:
                    print(f"❌ Error generando embedding texto en CPU fallback: {cpu_err}")
                    cleanup_memory()
                    return None
            else:
                print(f"❌ Error generando embedding texto: {e}")
                cleanup_memory()
                return None
        except Exception as e:
            print(f"❌ Error generando embedding texto: {e}")
            cleanup_memory()
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

        try:
            from qdrant_client.models import PayloadSchemaType
            await client.create_payload_index(
                collection_name=self.content_mv_collection,
                field_name="tipo",
                field_schema=PayloadSchemaType.KEYWORD
            )
            await client.create_payload_index(
                collection_name=self.content_fde_collection,
                field_name="tipo",
                field_schema=PayloadSchemaType.KEYWORD
            )
            # Índice integer para filtrar imágenes por número de página
            await client.create_payload_index(
                collection_name=self.content_mv_collection,
                field_name="numero_pagina",
                field_schema=PayloadSchemaType.INTEGER
            )
            await client.create_payload_index(
                collection_name=self.content_fde_collection,
                field_name="numero_pagina",
                field_schema=PayloadSchemaType.INTEGER
            )
            # Índice keyword para filtrar por nombre de archivo
            await client.create_payload_index(
                collection_name=self.content_mv_collection,
                field_name="nombre_archivo",
                field_schema=PayloadSchemaType.KEYWORD
            )
            await client.create_payload_index(
                collection_name=self.content_fde_collection,
                field_name="nombre_archivo",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            # Si ya existe, Qdrant podría lanzar un error o ignorarlo silenciosamente según la versión
            pass

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
        figuras_filtro: List[str] = None,
        filtro_tipo: str = None,
        filtro_paginas: List[int] = None
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
            from qdrant_client.models import MatchAny, MatchValue, Filter, FieldCondition, HasIdCondition
            
            must_conditions = []
            should_conditions = []
            
            if filtro_tipo:
                must_conditions.append(FieldCondition(key="tipo", match=MatchValue(value=filtro_tipo)))
                
            if figuras_filtro:
                should_conditions.append(FieldCondition(key="figuras", match=MatchAny(any=figuras_filtro)))
                
            if filtro_paginas:
                must_conditions.append(FieldCondition(key="numero_pagina", match=MatchAny(any=filtro_paginas)))
                
            if must_conditions or should_conditions:
                qdrant_filter = Filter(
                    must=must_conditions if must_conditions else None,
                    should=should_conditions if should_conditions else None
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
                limit=top_k * 2,  # Traemos un poco más para filtrar después
                with_vectors=True
            )

            # Filtrar y rankear candidatos
            todos_candidatos = []
            for r in mv_response.points:
                score = float(r.score)
                tiene_figura_exacta = False
                payload = r.payload or {}
                if "imagen_path" in payload:
                    payload = dict(payload)
                    payload["imagen_path"] = normalizar_ruta_imagen(payload["imagen_path"])
                if figuras_filtro and "figuras" in payload:
                    if any(f in payload["figuras"] for f in figuras_filtro):
                        tiene_figura_exacta = True
                todos_candidatos.append({
                    "id": r.id,
                    "score": score,
                    "payload": payload,
                    "tiene_figura_exacta": tiene_figura_exacta,
                    "vector": r.vector
                })

            if not todos_candidatos:
                return [], False

            todos_candidatos.sort(key=lambda x: x['score'], reverse=True)
            mejor_score = todos_candidatos[0]['score']

            umbral = min_score

            print(f"   🕵️ Reranking {len(todos_candidatos)} candidatos | "
                  f"top-1={mejor_score:.4f} | umbral={umbral:.4f} "
                  f"({'absoluto dinámico' if umbral > 0.0 else 'top-k puro'})")

            resultados = []
            descartados = 0
            for c in todos_candidatos:
                if c['score'] >= umbral or c['tiene_figura_exacta']:
                    resultados.append({
                        "id": c['id'],
                        "score": c['score'],
                        "payload": c['payload'],
                        "vector": c['vector']
                    })
                else:
                    descartados += 1
                    payload = c['payload']
                    print(f"      ❌ RECHAZADO: score={c['score']:.4f} "
                          f"| tipo={payload.get('tipo','?')} "
                          f"| pág={payload.get('numero_pagina','?')}")

            if descartados > 0:
                print(f"      🗑️ {descartados} candidatos rechazados (score < {umbral:.4f})")

            resultados = resultados[:top_k]

            if not resultados:
                has_rejected = umbral > 0.0  # Solo "rejected" si había un umbral activo

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
# FUNCIONES PURAS DE CLASIFICACIÓN Y FILTRADO
# ============================================================================

# Palabras clave que indican solicitud explícita de imagen
_KEYWORDS_IMAGEN = [
    "mostrá", "mostrar", "muéstrame",
    "enseñame", "quiero ver una", "ver imagen",
    "ver foto", "ver figura", "dame una imagen",
    "buscá una imagen", "buscar una imagen"
]

# Patrón compilado: case-insensitive
_PATRON_IMAGEN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _KEYWORDS_IMAGEN) + r")\b",
    re.IGNORECASE,
)


def detectar_intencion_imagen(texto: str) -> bool:
    """Retorna True si el texto contiene al menos una palabra clave o frase fuerte de solicitud de imagen.

    Usa matching case-insensitive para evitar falsos positivos con palabras sueltas como 'ver' o 'imagen'.
    """
    return bool(_PATRON_IMAGEN.search(texto))


def filtrar_resultados_busqueda(
    resultados: List[Dict],
    requiere_imagen: bool,
    tiene_imagen_adjunta: bool,
) -> Tuple[List[Dict], List[str]]:
    """Filtra resultados de búsqueda según el tipo de consulta.

    Cuando ``requiere_imagen=False`` y ``tiene_imagen_adjunta=False``, excluye
    todos los resultados de tipo ``"imagen"`` y retorna ``imagenes_relevantes=[]``.

    Cuando se incluyen imágenes, no limita la cantidad.

    Returns:
        Tupla ``(resultados_filtrados, imagenes_relevantes)`` donde
        ``imagenes_relevantes`` contiene los ``imagen_path`` de los resultados
        de imagen incluidos.
    """
    if not requiere_imagen and not tiene_imagen_adjunta:
        filtrados = [r for r in resultados if r.get("payload", {}).get("tipo") != "imagen"]
        return filtrados, []

    # Incluir todas las imágenes que rankearon
    filtrados: List[Dict] = []
    imagenes_relevantes: List[str] = []

    for r in resultados:
        payload = r.get("payload", {})
        if payload.get("tipo") == "imagen":
            filtrados.append(r)
            ruta = payload.get("imagen_path")
            if ruta:
                imagenes_relevantes.append(ruta)
        else:
            filtrados.append(r)

    return filtrados, imagenes_relevantes


def extraer_paginas_de_resultados(resultados: List[Dict]) -> List[int]:
    """Extrae números de página únicos de resultados de tipo ``"texto"``.

    Preserva el orden de primera aparición y elimina duplicados.
    """
    vistas: set[int] = set()
    paginas: List[int] = []
    for r in resultados:
        payload = r.get("payload", {})
        if payload.get("tipo") != "texto":
            continue
        pagina = payload.get("numero_pagina")
        if pagina is not None and pagina not in vistas:
            vistas.add(pagina)
            paginas.append(pagina)
    return paginas


def rerank_imagenes_por_caption(
    query_embedding: np.ndarray,
    candidatas: List[Dict],
    umbral: float = 0.45,
) -> List[Dict]:
    """Re-rankea imágenes candidatas por similitud MaxSim (Late Interaction) con la consulta.

    Cada candidata debe tener una clave ``caption_embedding`` con el
    multi-vector embedding (array 2-D).
    """
    if len(candidatas) == 0:
        return []

    q = np.asarray(query_embedding, dtype=np.float64)

    scored: List[Tuple[float, Dict]] = []
    for cand in candidatas:
        emb = cand.get("caption_embedding")
        if emb is None:
            continue
        c = np.asarray(emb, dtype=np.float64)
        
        # MaxSim (Late Interaction)
        if q.ndim == 2 and c.ndim == 2:
            sim_matrix = np.dot(q, c.T)
            sim = float(np.sum(np.max(sim_matrix, axis=1)))
        else:
            q_mean = q.mean(axis=0) if q.ndim == 2 else q
            c_mean = c.mean(axis=0) if c.ndim == 2 else c
            q_norm = np.linalg.norm(q_mean)
            c_norm = np.linalg.norm(c_mean)
            q_mean = q_mean / q_norm if q_norm > 0 else q_mean
            c_mean = c_mean / c_norm if c_norm > 0 else c_mean
            sim = float(np.dot(q_mean, c_mean))

        img_name = os.path.basename(cand.get("payload", {}).get("imagen_path", "?"))
        if sim >= umbral:
            scored.append((sim, cand))
            print(f"      📊 Rerank caption: {img_name} → sim={sim:.4f} ✓")
        else:
            print(f"      📊 Rerank caption: {img_name} → sim={sim:.4f} ✗ (< {umbral})")

    # Ordenar por similitud descendente
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Filtro relativo: solo mantener candidatas dentro del 75% del top score
    if scored:
        top_sim = scored[0][0]
        rel_cutoff = top_sim * 0.75
        before = len(scored)
        scored = [(s, c) for s, c in scored if s >= rel_cutoff]
        if len(scored) < before:
            print(f"      🔻 Filtro relativo: {before} → {len(scored)} (cutoff={rel_cutoff:.4f}, top={top_sim:.4f})")
    
    return [cand for _, cand in scored]


# ============================================================================
# ESTADO DEL GRAFO LANGGRAPH
# ============================================================================

class AgentState(TypedDict):
    """Estado del sistema de agentes"""
    messages: Annotated[list, add_messages]
    consulta_usuario: str
    consulta_resuelta: str
    imagen_consulta: Optional[str]
    contexto_memoria: str
    ontologia: Dict
    contexto_ontologico: str
    clasificacion: str
    requiere_imagen: bool
    consulta_optimizada: str
    filtros_ontologia: List[str]
    resultados_busqueda: List[Dict[str, Any]]
    contexto_documentos: str
    imagenes_relevantes: List[Any]
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

    async def _llm_ainvoke(self, messages: List[Any]) -> Any:
        """
        Invoca a Gemini con retry y rotación de API key ante rate limits.
        """
        intentos = 5
        delay = 2.0
        for intento in range(intentos):
            try:
                response = await self.llm.ainvoke(messages)
                return response
            except Exception as e:
                err_str = str(e)
                es_rate_limit = _is_quota_error(e)
                if es_rate_limit and intento < intentos - 1:
                    # Rotar key y re-crear LLM
                    old_key = getattr(self.llm, 'google_api_key', '') or ''
                    if old_key:
                        google_key_rotator.report_failure(old_key)
                    self.llm = create_google_llm(
                        model="gemini-2.5-flash",
                        temperature=0,
                        max_output_tokens=8192
                    )
                    self.llm_text = self.llm
                    self.llm_vision = self.llm

                    print(f"⚠️ [Gemini Rate Limit] Key rotada. Reintentando inmediatamente... (Intento {intento+1}/{intentos})")
                    await asyncio.sleep(0.1)
                else:
                    raise e

    def inicializar_componentes(self):
        """Inicializar todos los componentes"""
        if getattr(self, "_componentes_inicializados", False):
            return

        print("\n" + "="*80)
        print("🚀 SISTEMA RAG HISTOPATOLOGÍA - ColPali PURO + MUVERA + LangGraph")
        print("="*80)

        # LLM (con key rotativa)
        self.llm = create_google_llm(
            model="gemini-2.5-flash",
            temperature=0,
            max_output_tokens=8192
        )
        self.llm_text = self.llm
        self.llm_vision = self.llm

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

        # Extractor de ontología (con rotación de keys)
        self.extractor_ontologia = ExtractorOntologia()
        self.ontologia = self.extractor_ontologia.cargar_ontologia()

        # LangGraph
        self._inicializar_langgraph()
        self._componentes_inicializados = True
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
            
        # Resolución de correferencias y referencias contextuales
        consulta_resuelta = state["consulta_usuario"]
        if history and history.strip():
            try:
                print("   🔍 Resolviendo referencias contextuales de la consulta...")
                prompt_resolucion = f"""Eres un experto en histopatología y lingüística. Tu tarea es resolver correferencias y referencias contextuales en la consulta actual del usuario utilizando el historial de conversación provisto.

Si la consulta del usuario hace referencia a elementos anteriores (como 'la imagen', 'el tejido anterior', 'esta célula', 'el mismo órgano', 'él', 'ella', 'el de antes', 'la anterior', etc.), debes reescribir la consulta reemplazando esas referencias con los términos específicos mencionados en el historial (por ejemplo, 'bazo', 'espermátide tardía', 'arteria muscular', etc.).
Si la consulta no contiene ninguna referencia contextual o el historial está vacío, debes devolver la consulta original exactamente igual.

Historial de conversación:
{history}

Consulta actual del usuario: {state["consulta_usuario"]}

Responde ÚNICAMENTE con la consulta reescrita, sin explicaciones, introducciones ni comentarios adicionales."""

                messages = [
                    SystemMessage(content="Eres un asistente que reescribe consultas para resolver correferencias basándose en el historial de chat. Tu respuesta debe ser estrictamente la consulta resuelta, nada más."),
                    HumanMessage(content=prompt_resolucion)
                ]
                resolucion_response = await self._llm_ainvoke(messages)
                resolved = resolucion_response.content.strip()
                if resolved:
                    if (resolved.startswith('"') and resolved.endswith('"')) or (resolved.startswith("'") and resolved.endswith("'")):
                        resolved = resolved[1:-1].strip()
                    consulta_resuelta = resolved
                    print(f"   🎯 Consulta original: '{state['consulta_usuario']}'")
                    print(f"   🎯 Consulta resuelta:   '{consulta_resuelta}'")
            except Exception as e:
                print(f"   ⚠️ Error resolviendo referencias: {e}")
        
        state["consulta_resuelta"] = consulta_resuelta
        state["trayectoria"].append({"nodo": "inicializar", "timestamp": time.time()})
        return state

    async def _nodo_analizar_ontologia(self, state: AgentState) -> AgentState:
        if not state["ontologia"]:
            state["contexto_ontologico"] = "No disponible"
            state["filtros_ontologia"] = []
        else:
            terminos = self.extractor_ontologia.buscar_en_ontologia(state["consulta_resuelta"], state["ontologia"])
            state["contexto_ontologico"] = "\n".join(terminos)
            state["filtros_ontologia"] = [t.split(":")[1].strip() for t in terminos[:3]] if terminos else []

        state["trayectoria"].append({"nodo": "analizar_ontologia", "timestamp": time.time()})
        return state

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        # Priority 1: Image upload override
        imagen_upload = (
            state.get('imagen_consulta')
            and os.path.exists(state['imagen_consulta'])
        )

        info_imagen = f"\nImagen adjunta: Sí" if state.get('imagen_consulta') else "\nImagen adjunta: No"
        messages = [
            SystemMessage(content="""Eres un experto en histopatología. Clasifica consultas.
Debes determinar la intención del usuario. SOLAMENTE si el usuario pide EXPLÍCITAMENTE que se le muestre, busque o proporcione una imagen, foto, figura o micrografía (ej: "mostrame una foto", "quiero ver una imagen de..."), debes indicarlo.
Si el usuario hace una pregunta teórica, o usa palabras como "ver" o "imagen" sin estar pidiendo explícitamente ver una imagen (ej: "qué se puede ver en la muestra", "qué es una imagen digital"), debes indicarlo como FALSE.
Termina tu respuesta EXACTAMENTE con la línea "REQUIERE_IMAGEN: TRUE" si el usuario desea que le muestres una imagen, o "REQUIERE_IMAGEN: FALSE" si es solo una pregunta o comentario."""),
            HumanMessage(content=f"CONSULTA: {state['consulta_usuario']}{info_imagen}\nCONTEXTO ONTOLÓGICO:\n{state['contexto_ontologico']}")
        ]
        response = await self._llm_ainvoke(messages)
        state["clasificacion"] = response.content

        # Determine requiere_imagen using priority chain:
        #   Priority 1: Image upload present → True
        #   Priority 2: LLM says REQUIERE_IMAGEN: TRUE/FALSE → use that
        #   Priority 3: Fallback to detectar_intencion_imagen(consulta_usuario)
        if imagen_upload:
            state["requiere_imagen"] = True
        elif "REQUIERE_IMAGEN: TRUE" in response.content.upper():
            state["requiere_imagen"] = True
        elif "REQUIERE_IMAGEN: FALSE" in response.content.upper():
            state["requiere_imagen"] = False
        else:
            state["requiere_imagen"] = detectar_intencion_imagen(state['consulta_usuario'])

        state["trayectoria"].append({"nodo": "clasificar", "timestamp": time.time()})
        return state

    async def _nodo_optimizar_consulta(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content="""Eres un optimizador de consultas para un sistema RAG de histopatología.
Tu ÚNICA tarea es reformular la consulta del usuario en términos de búsqueda precisos.

REGLAS ESTRICTAS:
1. Responde SOLAMENTE con la consulta optimizada, SIN explicaciones ni texto adicional.
2. Enfócate EXCLUSIVAMENTE en el tema de la consulta actual.
3. Usa terminología histológica precisa cuando sea apropiado.
4. NO incluyas frases como "Aquí está la consulta optimizada" o "Sugiero buscar".
5. La salida debe ser SOLO los términos de búsqueda, nada más.
6. Si la consulta pide imágenes, enfócate en el TEMA (tejido/órgano/estructura), no en la palabra "imagen".

Ejemplo:
- Consulta: "mostrame imagenes de arterias" → "arterias histología corte transversal túnica íntima media adventicia"
- Consulta: "qué es el epitelio estratificado" → "epitelio estratificado clasificación características capas celulares"
"""),
            HumanMessage(content=f"CONSULTA: {state['consulta_resuelta']}\nCONTEXTO ONTOLÓGICO: {state['contexto_ontologico'][:500]}")
        ]
        response = await self._llm_ainvoke(messages)
        state["consulta_optimizada"] = response.content.strip()
        print(f"   🔧 Consulta optimizada: {state['consulta_optimizada'][:200]}")
        state["trayectoria"].append({"nodo": "optimizar_consulta", "timestamp": time.time()})
        return state

    def _calcular_dhash(self, image_path: str, hash_size: int = 16) -> Optional[np.ndarray]:
        """
        Calcula el Difference Hash (dHash) de una imagen.
        
        dHash es robusto a cambios de resolución, compresión JPEG,
        ajustes de brillo/contraste y re-encoding.
        Compara píxeles adyacentes horizontalmente → patrón de gradientes.
        
        Args:
            image_path: ruta a la imagen
            hash_size: tamaño del hash (hash_size x hash_size bits)
        Returns:
            np.ndarray de bools (hash_size * hash_size bits) o None si falla
        """
        try:
            img = Image.open(image_path).convert("L")  # Escala de grises
            # Redimensionar a (hash_size+1, hash_size) para comparar adyacentes
            img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.float32)
            # dHash: pixel[x] > pixel[x+1] para cada fila
            return (pixels[:, 1:] > pixels[:, :-1]).flatten()
        except Exception as e:
            print(f"   ⚠️ Error calculando dHash: {e}")
            return None

    def _verificar_match_visual(self, query_path: str, match_path: str) -> float:
        """
        Compara dos imágenes usando dHash (perceptual hash).
        
        Resultados típicos:
            Misma imagen (diferente compresión/resolución): >0.90
            Imágenes H&E diferentes: ~0.45-0.55
            Imágenes completamente distintas: <0.40
        
        Returns:
            float: Similitud entre 0.0 y 1.0
        """
        hash1 = self._calcular_dhash(query_path)
        hash2 = self._calcular_dhash(match_path)
        
        if hash1 is None or hash2 is None:
            return 1.0  # Asumir match si falla
        
        # Similitud = 1 - (distancia Hamming normalizada)
        hamming_distance = np.sum(hash1 != hash2)
        similarity = 1.0 - (hamming_distance / len(hash1))
        return float(similarity)

    async def _nodo_buscar(self, state: AgentState) -> AgentState:
        resultados = []
        has_rejected = False
        state["abortar_reset"] = False  # Default

        print(f"   📝 Consulta original: {state['consulta_usuario'][:150]}")
        print(f"   📝 Consulta optimizada: {state['consulta_optimizada'][:200]}")

        requiere_imagen = state.get('requiere_imagen', False)
        tiene_imagen_adjunta = (
            bool(state.get('imagen_consulta'))
            and os.path.exists(state['imagen_consulta'])
        )

        # ── PATH 3: Consulta_Imagen_Upload ──────────────────────────────
        if tiene_imagen_adjunta:
            print("\n🔍 [Path 3 — Consulta_Imagen_Upload] Búsqueda con embedding de imagen")
            query_mv = self.procesador.generar_embedding_imagen(state['imagen_consulta'])
            # Queremos buscar sin umbral estricto inicial, para que llegue a verificación
            umbral_busqueda = 0.0

            if query_mv is not None:
                t0 = time.time()
                query_fde = self.procesador.generar_query_muvera(query_mv)
                figuras_en_consulta = self._extraer_figuras_de_texto(state['consulta_optimizada'])
                t1 = time.time()

                resultados, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                    query_mv, query_fde,
                    min_score=umbral_busqueda,
                    figuras_filtro=figuras_en_consulta,
                    filtro_tipo="imagen",
                )
                t2 = time.time()
                print(f"⏱️ Tiempos: FDE={(t1-t0):.2f}s | Búsqueda+Rerank={(t2-t1):.2f}s")

                # Filtrar: limitar a 3 imágenes
                resultados, _ = filtrar_resultados_busqueda(
                    resultados, requiere_imagen=True, tiene_imagen_adjunta=True,
                )

                # ── Verificación por embeddings (embedding + dHash) ──
                # Verifica CADA imagen individualmente contra la imagen del usuario.
                try:
                    UMBRAL_VERIFICACION = float(os.getenv("VERIFICATION_THRESHOLD", "830"))
                except (ValueError, TypeError):
                    print("⚠️ VERIFICATION_THRESHOLD inválido, usando default 830")
                    UMBRAL_VERIFICACION = 830.0


                imagenes_a_verificar = [
                    r for r in resultados
                    if r.get('payload', {}).get('tipo') == 'imagen'
                ]

                ids_rechazados = set()
                for img_result in imagenes_a_verificar:
                    match_path = img_result['payload'].get('imagen_path', '')
                    if not match_path or not os.path.exists(match_path):
                        continue

                    # Intentar obtener el vector pre-calculado para evitar recalculación y OOM
                    match_mv = img_result.get("vector")
                    if match_mv is not None:
                        match_mv = np.array(match_mv, dtype=np.float32)
                    else:
                        match_mv = self.procesador.generar_embedding_imagen(match_path)
                    
                    if match_mv is None:
                        continue

                    sim_matrix = np.dot(query_mv, match_mv.T)
                    maxsim_directo = float(np.sum(np.max(sim_matrix, axis=1)))

                    match_name = os.path.basename(match_path)
                    qdrant_score = img_result.get('score', 0.0)

                    print(f"\n   🔬 VERIFICACIÓN EMBEDDINGS: query vs {match_name}")
                    print(f"      MaxSim directo (mismo modelo): {maxsim_directo:.2f}")
                    print(f"      Score Qdrant (índice):         {qdrant_score:.2f}")
                    print(f"      Umbral verificación:           {UMBRAL_VERIFICACION:.2f}")

                    if maxsim_directo < UMBRAL_VERIFICACION:
                        print(f"      ❌ Tejido NO coincide semánticamente → score bajo (score: {maxsim_directo:.2f})")
                        ids_rechazados.add(img_result['id'])
                    else:
                        # SIEMPRE verificar visualmente con dHash, sin importar el score.
                        # Imágenes histológicas distintas pueden tener MaxSim alto (>900)
                        # porque comparten estructuras celulares similares.
                        # Solo el dHash confirma que es la MISMA imagen.
                        print(f"      ✅ Tejido coincide semánticamente (score: {maxsim_directo:.2f}). Verificando visualmente...")
                        dhash_sim = self._verificar_match_visual(state['imagen_consulta'], match_path)
                        DHASH_THRESHOLD = 0.80
                        print(f"         Similitud visual (dHash): {dhash_sim:.4f} (umbral: {DHASH_THRESHOLD})")

                        if dhash_sim < DHASH_THRESHOLD:
                            print(f"         ❌ RECHAZADO: No es la misma imagen (dHash {dhash_sim:.4f} < {DHASH_THRESHOLD})")
                            ids_rechazados.add(img_result['id'])
                        else:
                            print(f"         ✅ Match visual confirmado — imagen idéntica en base de datos")

                if ids_rechazados:
                    resultados = [r for r in resultados if r.get('id') not in ids_rechazados]
                    print(f"      🗑️ {len(ids_rechazados)} imagen(es) rechazada(s) por verificación visual")
                    if len([r for r in resultados if r.get('payload', {}).get('tipo') == 'imagen']) == 0:
                        print("      🛑 Todas las imágenes fueron rechazadas por los umbrales de seguridad.")
                        has_rejected = True
                
                # En el caso de subida de imagen, si no hay imágenes válidas, debe fallar.
                if len([r for r in resultados if r.get('payload', {}).get('tipo') == 'imagen']) == 0:
                    has_rejected = True

        # ── PATH 2: Consulta_Imagen_Texto ───────────────────────────────
        # Estrategia: usar MUVERA para identificar el documento relevante,
        # luego buscar en TODOS los chunks de texto de ese documento las
        # etiquetas "Imagen N: descripción" y asociarlas con las imágenes
        # de la misma página.
        elif requiere_imagen:
            print("\n🔍 [Path 2 — Consulta_Imagen_Texto] Búsqueda de texto + imágenes por etiqueta semántica")
            query_mv = self.procesador.generar_embedding_texto(state['consulta_optimizada'])

            if query_mv is not None:
                t0 = time.time()
                query_fde = self.procesador.generar_query_muvera(query_mv)
                figuras_en_consulta = self._extraer_figuras_de_texto(state['consulta_optimizada'])
                t1 = time.time()

                # Paso 1: Buscar texto semánticamente similar (para contexto del LLM)
                resultados_texto, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                    query_mv, query_fde,
                    min_score=0.0,
                    figuras_filtro=figuras_en_consulta,
                    filtro_tipo="texto"
                )
                t2 = time.time()
                print(f"⏱️ Tiempos: FDE={(t1-t0):.2f}s | Búsqueda+Rerank={(t2-t1):.2f}s")

                # Paso 2: Identificar el documento principal de los resultados
                doc_scores = {}
                for r in resultados_texto:
                    payload = r.get('payload', {})
                    doc = payload.get('nombre_archivo', '')
                    score = r.get('score', 0.0)
                    if doc:
                        doc_scores[doc] = doc_scores.get(doc, 0.0) + score
                # Documento con mayor score acumulado
                doc_principal = max(doc_scores, key=doc_scores.get) if doc_scores else ''
                print(f"   📖 Documento principal: {doc_principal} (score acumulado: {doc_scores.get(doc_principal, 0):.2f})")

                # Paso 3: Scroll de TODOS los chunks de texto del documento principal
                # para buscar etiquetas "Imagen N: descripción"
                paginas_con_etiqueta = {}  # página -> lista de descripciones de etiqueta
                imagenes_encontradas = []
                try:
                    client = self.gestor_qdrant.client
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    if doc_principal:
                        scroll_texto = await client.scroll(
                            collection_name=self.gestor_qdrant.content_mv_collection,
                            scroll_filter=Filter(
                                must=[
                                    FieldCondition(key="tipo", match=MatchValue(value="texto")),
                                    FieldCondition(key="nombre_archivo", match=MatchValue(value=doc_principal)),
                                ]
                            ),
                            limit=1000,
                            with_payload=True,
                            with_vectors=False,
                        )
                        chunks_doc = scroll_texto[0] if scroll_texto else []
                        print(f"   📄 {len(chunks_doc)} chunks de texto en {doc_principal}")

                        # Buscar etiquetas "Imagen N: descripción" en cada chunk
                        for chunk in chunks_doc:
                            cp = chunk.payload or {}
                            texto_chunk = cp.get('texto', '')
                            pg_chunk = cp.get('numero_pagina')
                            # Capturar "Imagen 19: Espermatozoides", "Imagen 6: Corte transversal", etc.
                            matches = re.findall(r'[Ii]magen\s+(\d+(?:[\.\-·]\d+)?)\s*:\s*([^\n]{1,100})', texto_chunk)
                            for num, desc in matches:
                                if pg_chunk is not None:
                                    paginas_con_etiqueta.setdefault(pg_chunk, []).append({
                                        'numero': num,
                                        'descripcion': desc.strip(),
                                    })
                        if paginas_con_etiqueta:
                            print(f"   🏷️ Etiquetas encontradas en {doc_principal}:")
                            for pg, etiquetas in sorted(paginas_con_etiqueta.items()):
                                for et in etiquetas:
                                    print(f"      Pg {pg}: Imagen {et['numero']}: {et['descripcion']}")
                except Exception as e:
                    print(f"   ⚠️ Error buscando etiquetas de texto: {e}")

                # Paso 4: Rerank las etiquetas por similitud semántica con la consulta
                # Esto determina CUÁLES imágenes son relevantes para la consulta
                # IMPORTANTE: Usar embedding de la consulta ORIGINAL del usuario para el rerank,
                # no la consulta optimizada, para evitar contaminación del optimizador LLM.
                query_mv_para_rerank = self.procesador.generar_embedding_texto(state['consulta_resuelta'])
                if query_mv_para_rerank is None:
                    query_mv_para_rerank = query_mv  # Fallback a la optimizada
                else:
                    print(f"   🎯 Usando embedding de consulta resuelta para rerank de etiquetas")

                etiquetas_rankeadas = []
                if paginas_con_etiqueta:
                    for pg, etiquetas in paginas_con_etiqueta.items():
                        for et in etiquetas:
                            texto_etiqueta = f"Imagen {et['numero']}: {et['descripcion']}"
                            emb = self.procesador.generar_embedding_texto(texto_etiqueta)
                            if emb is not None:
                                # Calcular similitud usando MaxSim (Late Interaction) de ColPali
                                q = np.asarray(query_mv_para_rerank, dtype=np.float64)
                                c = np.asarray(emb, dtype=np.float64)
                                if q.ndim == 2 and c.ndim == 2:
                                    sim_matrix = np.dot(q, c.T)
                                    sim = float(np.sum(np.max(sim_matrix, axis=1)))
                                else:
                                    q_mean = q.mean(axis=0) if q.ndim == 2 else q
                                    c_mean = c.mean(axis=0) if c.ndim == 2 else c
                                    q_norm = np.linalg.norm(q_mean)
                                    c_norm = np.linalg.norm(c_mean)
                                    q_mean = q_mean / q_norm if q_norm > 0 else q_mean
                                    c_mean = c_mean / c_norm if c_norm > 0 else c_mean
                                    sim = float(np.dot(q_mean, c_mean))

                                etiquetas_rankeadas.append({
                                    'pagina': pg,
                                    'numero': et['numero'],
                                    'descripcion': et['descripcion'],
                                    'similitud': sim,
                                })
                                print(f"      📊 Imagen {et['numero']}: {et['descripcion'][:50]} → sim={sim:.4f}")

                    # Ordenar por similitud descendente
                    etiquetas_rankeadas.sort(key=lambda x: x['similitud'], reverse=True)

                # Paso 5: Scroll de TODAS las imágenes y asociar con etiquetas
                all_image_points = []
                try:
                    scroll_result = await client.scroll(
                        collection_name=self.gestor_qdrant.content_mv_collection,
                        scroll_filter=Filter(
                            must=[FieldCondition(key="tipo", match=MatchValue(value="imagen"))]
                        ),
                        limit=1000,
                        with_payload=True,
                        with_vectors=False,
                    )
                    all_image_points = scroll_result[0] if scroll_result else []
                    for punto in all_image_points:
                        if punto.payload and "imagen_path" in punto.payload:
                            punto.payload["imagen_path"] = normalizar_ruta_imagen(punto.payload["imagen_path"])
                except Exception as e:
                    print(f"   ⚠️ Error obteniendo imágenes de Qdrant: {e}")

                # Paso 6: Para las top 3 etiquetas, buscar la imagen en la misma página
                paginas_usadas = set()
                if etiquetas_rankeadas:
                    for et in etiquetas_rankeadas[:3]:
                        pg_target = et['pagina']
                        if pg_target in paginas_usadas:
                            continue
                        # Buscar imagen del documento principal en esa página
                        for punto in all_image_points:
                            payload = punto.payload or {}
                            if (payload.get('nombre_archivo', '') == doc_principal
                                    and payload.get('numero_pagina') == pg_target):
                                img_path = payload.get('imagen_path', '')
                                if img_path and os.path.exists(img_path):
                                    desc = f"Imagen {et['numero']}: {et['descripcion']}"
                                    imagenes_encontradas.append({
                                        "path": img_path,
                                        "descripcion": desc[:300]
                                    })
                                    paginas_usadas.add(pg_target)
                                    print(f"      ✅ Imagen seleccionada: {os.path.basename(img_path)} (Imagen {et['numero']}: {et['descripcion'][:50]})")
                                    break

                # Paso 7: Fallback — si no se encontraron imágenes por etiqueta,
                # usar imágenes del documento principal rankeadas por caption
                if not imagenes_encontradas and all_image_points:
                    print(f"   🔄 Fallback: rerank por caption de imágenes de {doc_principal}")
                    candidatas_fallback = []
                    for punto in all_image_points:
                        payload = punto.payload or {}
                        if payload.get('nombre_archivo', '') != doc_principal:
                            continue
                        caption_directo = (payload.get('texto', '') or '').strip()
                        contexto_pagina = (payload.get('contexto_texto', '') or '').strip()
                        caption_combinado = f"{caption_directo} {contexto_pagina}".strip()
                        if not caption_combinado:
                            continue
                        emb = self.procesador.generar_embedding_texto(caption_combinado)
                        if emb is not None:
                            candidatas_fallback.append({
                                "id": punto.id,
                                "payload": payload,
                                "caption_embedding": emb,
                            })
                    if candidatas_fallback:
                        imagenes_reranked = rerank_imagenes_por_caption(query_mv, candidatas_fallback, umbral=0.0)
                        for r in imagenes_reranked[:1]:
                            img_path = r.get("payload", {}).get("imagen_path", "")
                            caption = r.get("payload", {}).get("texto", "") or r.get("payload", {}).get("contexto_texto", "")
                            if img_path and os.path.exists(img_path):
                                imagenes_encontradas.append({
                                    "path": img_path,
                                    "descripcion": caption[:300]
                                })
                                print(f"      ✅ Imagen seleccionada (fallback): {os.path.basename(img_path)}")

                if imagenes_encontradas:
                    state["imagenes_relevantes"] = imagenes_encontradas[:1]  # Solo la mejor
                    print(f"   📋 1 imagen relevante seleccionada (de {len(imagenes_encontradas)} candidatas).")
                else:
                    state["imagenes_relevantes"] = []
                    print(f"   📋 No se encontraron imágenes relevantes.")

                # Solo texto para el contexto del LLM
                resultados = resultados_texto

        # ── PATH 1: Consulta_Texto ──────────────────────────────────────
        else:
            print("\n🔍 [Path 1 — Consulta_Texto] Búsqueda de solo texto")
            query_mv = self.procesador.generar_embedding_texto(state['consulta_optimizada'])

            if query_mv is not None:
                t0 = time.time()
                query_fde = self.procesador.generar_query_muvera(query_mv)
                figuras_en_consulta = self._extraer_figuras_de_texto(state['consulta_optimizada'])
                t1 = time.time()

                resultados, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                    query_mv, query_fde,
                    min_score=0.0,
                    figuras_filtro=figuras_en_consulta,
                )
                t2 = time.time()
                print(f"⏱️ Tiempos: FDE={(t1-t0):.2f}s | Búsqueda+Rerank={(t2-t1):.2f}s")

                # Excluir TODAS las imágenes — garantizar imagenes_relevantes = []
                resultados, _ = filtrar_resultados_busqueda(
                    resultados, requiere_imagen=False, tiene_imagen_adjunta=False,
                )

        # ── Logging de resultados ───────────────────────────────────────
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

        # ── Actualizar estado ───────────────────────────────────────────
        state["resultados_busqueda"] = resultados
        state["abortar_reset"] = has_rejected

        if has_rejected:
            print("🚨 ALERTA: Candidatos rechazados detectados. Se abortará la generación para evitar errores de contexto excesivo.")
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
                    caption = r['payload'].get('texto', '')
                    figuras = r['payload'].get('figuras', [])
                    figuras_str = ", ".join(figuras) if figuras else "No identificadas"
                    if img_path:
                        imagenes.append({
                            "path": img_path,
                            "descripcion": caption or contexto_texto[:300]
                        })
                        contextos.append(f"[RESULTADO PRINCIPAL - IMAGEN - Score: {score:.2f} - Fuente: {pdf_name} (Pg {page_num})]\nArchivo: {os.path.basename(img_path)}\nFiguras en esta página: {figuras_str}\nTexto asociado a esta imagen: {contexto_texto[:600]}")

            state["contexto_documentos"] = "\n\n---\n\n".join(contextos)
            # Solo la imagen con mayor score
            if not state.get("imagenes_relevantes"):
                state["imagenes_relevantes"] = imagenes[:1]
            # Si Path 2 ya seleccionó una, no agregar más

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

    def _detectar_tipo_consulta(self, state: AgentState) -> str:
        """
        Detecta el tipo de consulta basándose en el estado.
        
        Args:
            state: Estado actual del agente
            
        Returns:
            'imagen' si hay contexto visual disponible, 'texto' en caso contrario
        """
        # Verificar si hay imagen de consulta válida del usuario
        if state.get('imagen_consulta'):
            if os.path.exists(state['imagen_consulta']):
                return 'imagen'
        
        # Si el usuario pidió explícitamente una imagen por texto y recuperamos alguna
        if state.get('requiere_imagen', False) and state.get('imagenes_relevantes'):
            return 'imagen'
        
        # Si no hay imagen del usuario ni pidió explícitamente una, es una consulta de texto
        return 'texto'

    def _generar_prompt_sistema(self, tipo_consulta: str) -> str:
        """
        Genera el prompt del sistema adaptado al tipo de consulta.
        
        Args:
            tipo_consulta: 'imagen', 'texto', o 'imagen_no_encontrada'
            
        Returns:
            String con el prompt del sistema
        """
        if tipo_consulta == 'imagen_no_encontrada':
            # Prompt para cuando el usuario pidió imágenes.
            # El LLM debe referenciar imágenes por etiqueta (ej: "Imagen 13.4")
            # basándose en las menciones que aparecen en el contexto textual.
            # En _nodo_finalizar se parsean esas referencias y se buscan
            # las imágenes correspondientes en Qdrant.
            return """Eres un profesor experto en histopatología con un estilo amigable y educativo. 
Tu función es ayudar a estudiantes a comprender conceptos de histopatología 
respondiendo sus preguntas de forma clara y accesible.

INSTRUCCIÓN IMPORTANTE SOBRE IMÁGENES:
El usuario solicitó ver imágenes. En el contexto textual hay referencias a 
figuras e imágenes del manual (ej: "Imagen 13.4", "Figura 15.1"). 
DEBES mencionar estas referencias en tu respuesta usando el formato exacto 
que aparece en el texto (ej: "Imagen 13.4: Sarcómera").
El sistema buscará automáticamente las imágenes correspondientes para mostrarlas.

Si el contexto NO menciona ninguna imagen o figura relevante, informa al 
usuario amablemente que no se encontraron imágenes para su consulta.

REGLAS DE PRECISIÓN:
1. Responde basándote en el contexto textual proporcionado.
2. Cuando menciones una imagen, usa el formato exacto del texto: "Imagen X.X" o "Figura X.X".
3. Si el contexto es insuficiente, responde honestamente.
4. Nunca inventes referencias a imágenes que no estén en el contexto.
5. Usa un tono conversacional pero mantén el rigor científico.

ESTRUCTURA DE RESPUESTA:
1. **Respuesta directa**: Responde la pregunta de forma clara y concisa.
2. **Imágenes relevantes**: Menciona las imágenes/figuras del contexto que ilustran el tema.
3. **Explicación**: Desarrolla los conceptos relevantes.
4. **Evidencia**: Cita las fuentes del contexto."""

        elif tipo_consulta == 'texto':
            # Prompt conversacional para consultas de solo texto
            return """Eres un profesor experto en histopatología con un estilo amigable y educativo. 
Tu función es ayudar a estudiantes a comprender conceptos de histopatología 
respondiendo sus preguntas de forma clara y accesible.

REGLAS DE PRECISIÓN:
1. Responde basándote en el contexto textual proporcionado.
2. Puedes realizar deducciones lógicas apoyadas en el texto del contexto, 
   citando qué parte te permite deducirlo.
3. Si el contexto es insuficiente, responde honestamente: 
   "No tengo suficiente información en mis fuentes para responder eso con 
   precisión. ¿Podrías reformular tu pregunta o darme más detalles sobre qué 
   aspecto específico te interesa?"
4. Nunca inventes información que no esté en el contexto.
5. Usa un tono conversacional pero mantén el rigor científico.

ESTRUCTURA DE RESPUESTA:
1. **Respuesta directa**: Responde la pregunta de forma clara y concisa.
2. **Explicación**: Desarrolla los conceptos relevantes.
3. **Evidencia**: Cita las fuentes del contexto que respaldan tu respuesta.
4. **Contexto adicional** (opcional): Información relacionada que pueda ser útil."""
        
        else:  # tipo_consulta == 'imagen'
            # Prompt conversacional para consultas con imagen
            return """Eres un profesor experto en histopatología con un estilo amigable y educativo. 
Tu función es ayudar a estudiantes a comprender conceptos de histopatología 
analizando imágenes y respondiendo sus preguntas de forma clara y accesible.

REGLA FUNDAMENTAL SOBRE IMÁGENES RECUPERADAS:
Las imágenes etiquetadas como [IMAGEN RECUPERADA] son el RESULTADO de una 
búsqueda por similitud en la base de datos. Estas imágenes YA PASARON un 
umbral de similitud alto y SON la mejor coincidencia encontrada. Por lo tanto:
- DEBES describir y analizar las imágenes recuperadas usando el texto asociado.
- Si el usuario subió una imagen (etiquetada como [IMAGEN DE CONSULTA DEL USUARIO]), 
  la imagen recuperada ES la coincidencia encontrada para esa consulta.
- NO rechaces una imagen recuperada diciendo que "no coincide" con la del usuario.
- Si una página contiene múltiples figuras, analiza visualmente la imagen 
  recuperada y compárala con las DESCRIPCIONES de cada figura en el texto 
  asociado para determinar cuál es.

REGLAS DE PRECISIÓN:
1. Responde basándote en el contexto proporcionado (imágenes recuperadas y 
   fragmentos de texto).
2. Puedes realizar deducciones lógicas apoyadas en el texto o imágenes del 
   contexto, citando qué parte te permite deducirlo.
3. Si el contexto es insuficiente, responde honestamente: 
   "No tengo suficiente información en mis fuentes para responder eso con 
   precisión. ¿Podrías reformular tu pregunta o darme más detalles sobre qué 
   aspecto específico te interesa?"
4. Nunca inventes información que no esté en el contexto.
5. Usa un tono conversacional pero mantén el rigor científico.

REGLAS SOBRE FIGURAS:
1. Si el usuario pregunta por una figura específica (ej: "Figura 14.3"), 
   SOLO describe esa figura usando el texto asociado.
2. Si la imagen recuperada contiene múltiples figuras, identifica y describe 
   SOLO la que el usuario solicitó.

ESTRUCTURA DE RESPUESTA:
1. **Imagen encontrada**: Indica qué imagen se recuperó y su figura correspondiente.
2. **Análisis Visual**: Describe qué se observa en la imagen según el texto asociado.
3. **Identificación**: Qué órgano/tejido/estructura se observa.
4. **Evidencia**: Integra lo que se ve en la imagen con lo que dice el texto."""

    def _filtrar_referencias_imagenes(self, contexto_memoria: str) -> str:
        """
        Filtra las referencias de imágenes del historial de conversación.
        
        Remueve líneas que contienen marcadores de imagen como:
        - [IMAGEN RECUPERADA N: filename]
        - [IMAGEN DE CONSULTA DEL USUARIO]
        
        Args:
            contexto_memoria: Historial de conversación que puede contener marcadores de imagen
            
        Returns:
            Historial filtrado sin marcadores de imagen, preservando el contenido textual
        """
        import re
        
        if not contexto_memoria:
            return contexto_memoria
        
        # Dividir en líneas para procesar cada una
        lineas = contexto_memoria.split('\n')
        lineas_filtradas = []
        
        for linea in lineas:
            # Filtrar líneas que contienen marcadores de imagen
            if '[IMAGEN RECUPERADA' in linea or '[IMAGEN DE CONSULTA DEL USUARIO]' in linea:
                continue
            lineas_filtradas.append(linea)
        
        # Reconstruir el texto preservando los saltos de línea
        return '\n'.join(lineas_filtradas)

    def _construir_mensaje_usuario(
        self, 
        state: AgentState, 
        tipo_consulta: str
    ) -> List[Dict[str, Any]]:
        """
        Construye el contenido del mensaje de usuario adaptado al tipo de consulta.
        
        Args:
            state: Estado actual del agente
            tipo_consulta: 'imagen', 'texto', o 'imagen_no_encontrada'
            
        Returns:
            Lista de partes del mensaje (texto e imágenes)
        """
        # Construir historial de conversación si existe
        historial = ""
        if state.get("contexto_memoria"):
            # Filtrar referencias de imágenes si es una consulta de texto o imagen_no_encontrada
            contexto_a_usar = state['contexto_memoria']
            if tipo_consulta in ('texto', 'imagen_no_encontrada'):
                contexto_a_usar = self._filtrar_referencias_imagenes(contexto_a_usar)
            
            historial = f"\n========================================\nHISTORIAL DE CONVERSACIÓN RELEVANTE:\n{contexto_a_usar}\n========================================\n"
        
        # Inicializar contenido del mensaje
        user_content = []
        
        if tipo_consulta in ('texto', 'imagen_no_encontrada'):
            # Mensaje para consultas de solo texto
            texto_mensaje = f"""{historial}CONSULTA DEL USUARIO: {state["consulta_usuario"]}

========================================
CONTEXTO RECUPERADO DE LA BASE DE DATOS
(Esta es la ÚNICA fuente de verdad para tu respuesta)
========================================

{state["contexto_documentos"][:10000]}

========================================

Responde basándote ÚNICAMENTE en el contexto de arriba."""
            
            user_content.append({
                "type": "text",
                "text": texto_mensaje
            })
        
        else:  # tipo_consulta == 'imagen'
            # Construir información sobre imágenes
            info_imagen = ""
            if state.get('imagen_consulta'):
                info_imagen = "\nNOTA: El usuario proporcionó una imagen para análisis. Las imágenes recuperadas de la base de datos son las MEJORES COINCIDENCIAS encontradas para esa imagen."
            if state["imagenes_relevantes"]:
                info_imagen += f"\nSe encontraron {len(state['imagenes_relevantes'])} imágenes coincidentes en la base de datos. DEBES describir estas imágenes usando el texto asociado."
            
            # Mensaje para consultas con imagen
            texto_mensaje = f"""{historial}CONSULTA DEL USUARIO: {state["consulta_usuario"]}
{info_imagen}

========================================
CONTEXTO RECUPERADO DE LA BASE DE DATOS
(Esta es la ÚNICA fuente de verdad para tu respuesta)
========================================

{state["contexto_documentos"][:4000]}

========================================

Responde basándote ÚNICAMENTE en el contexto de arriba y las IMÁGENES adjuntas."""
            
            user_content.append({
                "type": "text",
                "text": texto_mensaje
            })
            
            # Determinar límite de imágenes según restricción de API (máximo 2 imágenes totales en free tier para no superar el TPM)
            max_imagenes_recuperadas = 1 if (state.get('imagen_consulta') and os.path.exists(state['imagen_consulta'])) else 2
            imagenes_a_procesar = state["imagenes_relevantes"][:max_imagenes_recuperadas]
            
            if len(state["imagenes_relevantes"]) > max_imagenes_recuperadas:
                print(f"   ⚠️ Limitando a {max_imagenes_recuperadas} imágenes de contexto (de {len(state['imagenes_relevantes'])}) para evitar límite de tokens de Groq.")
            
            # Añadir imágenes recuperadas al mensaje
            imagenes_cargadas_exitosamente = 0
            for i, img_item in enumerate(imagenes_a_procesar):
                try:
                    # Soportar tanto dict (nuevo) como string (legacy)
                    img_path = img_item["path"] if isinstance(img_item, dict) else img_item
                    if os.path.exists(img_path):
                        image_data = redimensionar_y_codificar_imagen(img_path, max_dim=384)
                        
                        user_content.append({
                            "type": "text", 
                            "text": f"\n[IMAGEN RECUPERADA {i+1}: {os.path.basename(img_path)}]"
                        })
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        })
                        print(f"   🖼️ Adjuntando imagen al prompt: {os.path.basename(img_path)}")
                        imagenes_cargadas_exitosamente += 1
                    else:
                        print(f"   ⚠️ Imagen no existe: {img_path}")
                except Exception as e:
                    print(f"   ⚠️ Error cargando imagen {img_item}: {e}")
            
            # Si el usuario subió una imagen, también la adjuntamos
            if state.get('imagen_consulta') and os.path.exists(state['imagen_consulta']):
                try:
                    query_image_data = redimensionar_y_codificar_imagen(state['imagen_consulta'], max_dim=384)
                    user_content.append({
                        "type": "text",
                        "text": "\n[IMAGEN DE CONSULTA DEL USUARIO]"
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{query_image_data}"}
                    })
                    print(f"   🖼️ Adjuntando imagen de consulta del usuario")
                    imagenes_cargadas_exitosamente += 1
                except Exception as e:
                    print(f"   ⚠️ Error cargando imagen de consulta: {e}")
            
            # Si todas las imágenes fallaron, tratar como consulta de texto
            if imagenes_cargadas_exitosamente == 0:
                print(f"   ⚠️ Todas las imágenes fallaron al cargar. Tratando como consulta de texto.")
                # Reconstruir mensaje como consulta de texto
                user_content = []
                
                # NO filtrar referencias de imágenes en el fallback - preservar el historial original
                # porque esto sigue siendo una consulta de imagen (tipo_consulta == 'imagen')
                texto_mensaje = f"""{historial}CONSULTA DEL USUARIO: {state["consulta_usuario"]}

========================================
CONTEXTO RECUPERADO DE LA BASE DE DATOS
(Esta es la ÚNICA fuente de verdad para tu respuesta)
========================================

{state["contexto_documentos"][:10000]}

========================================

Responde basándote ÚNICAMENTE en el contexto de arriba."""
                
                user_content.append({
                    "type": "text",
                    "text": texto_mensaje
                })
        
        return user_content

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        """Nodo 6: Generar respuesta basada EXCLUSIVAMENTE en contexto recuperado"""
        print("\n💭 Generando respuesta basada en contexto recuperado...")

        # 1. Check for "images requested but not found" case
        requiere_imagen = state.get('requiere_imagen', False)
        imagenes_encontradas = len(state.get('imagenes_relevantes', [])) > 0
        tiene_imagen_consulta = state.get('imagen_consulta') and os.path.exists(state.get('imagen_consulta', ''))

        if requiere_imagen and not imagenes_encontradas and not tiene_imagen_consulta:
            # User asked for images but none were found and no image was uploaded
            tipo_consulta = 'imagen_no_encontrada'
            print("   📝 Tipo de consulta: imagen solicitada pero no encontrada")
        else:
            # Normal detection: text-only or image-with-results
            tipo_consulta = self._detectar_tipo_consulta(state)
            print(f"   📝 Tipo de consulta detectado: {tipo_consulta}")

        # 2. Validar contexto insuficiente (Requirement 5.1, 5.3)
        contexto = state.get("contexto_documentos", "")
        if len(contexto.strip()) < 50:
            print("   ⚠️ Contexto insuficiente detectado (<50 caracteres)")
            # El prompt ya guía al LLM a responder apropiadamente con mensaje amigable

        # 3. Generar prompt del sistema adaptado
        system_prompt = self._generar_prompt_sistema(tipo_consulta)

        # 4. Construir mensaje de usuario con contenido adaptado
        user_content = self._construir_mensaje_usuario(state, tipo_consulta)

        # 5. Crear mensajes para el LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        # 6. Invocar LLM y generar respuesta
        response = await self._llm_ainvoke(messages)
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
                resume_response = await self._llm_ainvoke(messages)
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
            r'[Ii]magen\s+(\d+)(?=\s*:)',
            r'IMAGEN\s+(\d+)(?=\s*:)',
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
                    "texto": item.get("caption", ""),
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

    async def procesar_consulta_estado(self, consulta: str, imagen_path: Optional[str] = None, imagen_base64: Optional[str] = None, user_id: str = "default") -> AgentState:
        initial_state = AgentState(
            messages=[], consulta_usuario=consulta, consulta_resuelta="", imagen_consulta=imagen_path,
            imagen_base64=imagen_base64,
            contexto_memoria="", ontologia=self.ontologia or {}, contexto_ontologico="",
            clasificacion="", requiere_imagen=False, consulta_optimizada="", filtros_ontologia=[],
            resultados_busqueda=[], contexto_documentos="", imagenes_relevantes=[],
            respuesta_final="", trayectoria=[], user_id=user_id, tiempo_inicio=time.time(),
            abortar_reset=False
        )
        config = {"configurable": {"thread_id": user_id}}
        final_state = await self.compiled_graph.ainvoke(initial_state, config=config)
        return final_state

    async def procesar_consulta(self, consulta: str, imagen_path: Optional[str] = None, imagen_base64: Optional[str] = None, user_id: str = "default") -> str:
        final_state = await self.procesar_consulta_estado(consulta, imagen_path, imagen_base64, user_id)
        return final_state["respuesta_final"]

    def cerrar(self):
        cleanup_memory()


# ============================================================================
# MEDICAL AGENT ADAPTER FOR A2A SDK
# ============================================================================

class MedicalAgent(SistemaRAGColPaliPuro):
    """Agente médico con Colpali local y fallback inteligente a búsquedas web"""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    SYSTEM_INSTRUCTION = (
        'Eres un médico especialista experimentado que analiza consultas médicas, '
        'imágenes histológicas y proporciona análisis profesionales.'
    )

    def __init__(self):
        super().__init__()
        # Tavily Search Tool is disabled
        self.tavily_tool = None

    @property
    def qdrant_client(self):
        self.inicializar_componentes()
        return self.gestor_qdrant.client

    def _inicializar_langgraph(self):
        """Inicializar grafo de agentes con fallback a internet"""
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(AgentState)

        graph.add_node("recepcionar_consulta", self._nodo_recepcionar_consulta)
        graph.add_node("inicializar", self._nodo_inicializar)
        graph.add_node("analizar_ontologia", self._nodo_analizar_ontologia)
        graph.add_node("clasificar", self._nodo_clasificar)
        graph.add_node("optimizar_consulta", self._nodo_optimizar_consulta)
        graph.add_node("buscar", self._nodo_buscar)
        graph.add_node("generar_respuesta", self._nodo_generar_respuesta)
        graph.add_node("fallback_internet", self._nodo_fallback_internet)
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
                "fallback": "fallback_internet",
                "reset": "reset"
            }
        )
        
        graph.add_edge("generar_respuesta", "finalizar")
        graph.add_edge("fallback_internet", "finalizar")
        graph.add_edge("reset", "finalizar")
        graph.add_edge("finalizar", END)

        self.compiled_graph = graph.compile()

    def _decidir_camino_tras_busqueda(self, state: AgentState) -> str:
        requiere_imagen = state.get("requiere_imagen", False)
        imagenes_encontradas = len(state.get("imagenes_relevantes", [])) > 0
        
        # Conmutar a fallback si falló la verificación o no hay coincidencia visual/textual local
        if state.get("abortar_reset", False) or not state.get("resultados_busqueda") or (requiere_imagen and not imagenes_encontradas):
            return "fallback"
        return "generar"

    async def _nodo_fallback_internet(self, state: AgentState) -> AgentState:
        """Nodo de Fallback: búsqueda en internet con Tavily al no encontrar coincidencias locales"""
        print("🌐 FALLBACK: Buscando en internet...")
        
        # 1. Analizar imagen visualmente con Llama-4-Scout si existe
        hallazgos_visuales = ""
        imagen_path = state.get("imagen_consulta")
        
        if imagen_path and os.path.exists(imagen_path):
            print("   📸 Extrayendo hallazgos de imagen para consulta web...")
            try:
                img_b64 = redimensionar_y_codificar_imagen(imagen_path, max_dim=384)
                
                content = [
                    {
                        "type": "text",
                        "text": "Describe detalladamente los hallazgos de esta imagen médica: tejidos, anomalías o patrones de interés clínico para buscar información en la web."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
                res = await self._llm_ainvoke([HumanMessage(content=content)])
                hallazgos_visuales = res.content
            except Exception as e:
                print(f"   ⚠️ Error de visión en fallback: {e}")
                hallazgos_visuales = "Error en análisis visual de la imagen."

        # 2. Generar query para Tavily
        system_query_prompt = """Eres un experto en búsqueda médica. Genera una consulta de búsqueda precisa y corta en inglés o español basada en los hallazgos de la imagen y la consulta. Responde SOLO con la query."""
        user_query_prompt = f"Consulta del usuario: {state['consulta_usuario']}\nHallazgos visuales: {hallazgos_visuales}"
        
        try:
            messages = [
                SystemMessage(content=system_query_prompt),
                HumanMessage(content=user_query_prompt)
            ]
            res = await self._llm_ainvoke(messages)
            search_query = res.content.strip()
        except Exception:
            search_query = state.get("consulta_optimizada") or state["consulta_usuario"]

        # 3. Realizar búsqueda en internet (Tavily deshabilitado)
        search_info = "Búsqueda en internet deshabilitada."

        # 4. Generar respuesta final de fallback
        system_response_prompt = """Eres un médico especialista que responde consultas.
Dado que la consulta o la imagen NO coinciden con los manuales locales de histopatología, realizaste una búsqueda en la web.
1. Aclara amablemente al usuario que no se encontró el caso en la base de datos local y que buscaste en internet.
2. Proporciona una explicación estructurada basada en los hallazgos visuales (si aplica) y la búsqueda web.
3. Agrega disclaimers obligatorios aclarando que esto no sustituye una consulta médica real.
4. Cita las fuentes (URLs) encontradas."""

        user_response_prompt = f"""CONSULTA DEL USUARIO: {state['consulta_usuario']}
HISTORIAL: {state.get('contexto_memoria', '')}
HALLAZGOS VISUALES: {hallazgos_visuales}
RESULTADOS DE BÚSQUEDA WEB:
{search_info}"""

        try:
            res = await self._llm_ainvoke([
                SystemMessage(content=system_response_prompt),
                HumanMessage(content=user_response_prompt)
            ])
            state["respuesta_final"] = res.content
        except Exception as e:
            state["respuesta_final"] = f"Error generando respuesta de fallback: {e}"

        state["imagenes_relevantes"] = []
        state["contexto_documentos"] = f"[Búsqueda en Internet]\n{search_info}"
        state["trayectoria"].append({"nodo": "fallback_internet", "timestamp": time.time()})
        return state

    async def invoke(self, query: str, context_id: str, images: list[dict] = None) -> str:
        """Compatibilidad con ejecución directa"""
        final_text = ""
        async for chunk in self.stream(query, context_id, images):
            if chunk.get('is_task_complete'):
                final_text = chunk.get('content', '')
        return final_text

    async def stream(self, query: str, context_id: str, images: list[dict] = None) -> AsyncIterable[dict[str, Any]]:
        """Streaming de eventos y respuestas para el Agent SDK UI"""
        self.inicializar_componentes()

        # Extraer imagen base64 si viene de la parte de entrada
        imagen_base64 = None
        if images and len(images) > 0:
            img = images[0]
            img_data = img.get('data') or img.get('bytes')
            if isinstance(img_data, bytes):
                imagen_base64 = base64.b64encode(img_data).decode('utf-8')
            elif isinstance(img_data, str):
                imagen_base64 = img_data

        initial_state = AgentState(
            messages=[],
            consulta_usuario=query,
            consulta_resuelta="",
            imagen_consulta=None,
            imagen_base64=imagen_base64,
            contexto_memoria="",
            ontologia=self.ontologia or {},
            contexto_ontologico="",
            clasificacion="",
            requiere_imagen=False,
            consulta_optimizada="",
            filtros_ontologia=[],
            resultados_busqueda=[],
            contexto_documentos="",
            imagenes_relevantes=[],
            respuesta_final="",
            trayectoria=[],
            user_id=context_id,
            tiempo_inicio=time.time(),
            abortar_reset=False
        )

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🏥 Recepcionando consulta médica...',
            'status': 'analyzing_images'
        }

        # 1. Recepcionar
        state = await self._nodo_recepcionar_consulta(initial_state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🧠 Inicializando memoria y contexto...',
            'status': 'analyzing_images'
        }

        # 2. Inicializar
        state = await self._nodo_inicializar(state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔬 Analizando ontología histológica...',
            'status': 'classifying'
        }

        # 3. Analizar Ontología
        state = await self._nodo_analizar_ontologia(state)

        # 4. Clasificar consulta
        state = await self._nodo_clasificar(state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔎 Optimizando consulta para búsqueda...',
            'status': 'searching'
        }

        # 5. Optimizar
        state = await self._nodo_optimizar_consulta(state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔍 Buscando en base de datos local (ColPali)...',
            'status': 'searching'
        }

        # 6. Buscar localmente
        state = await self._nodo_buscar(state)

        # 7. Decidir camino: Local RAG o Fallback
        camino = self._decidir_camino_tras_busqueda(state)

        if camino == "generar":
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '📝 Generando respuesta basada en el manual local...',
                'status': 'generating_response'
            }
            state = await self._nodo_generar_respuesta(state)
        elif camino == "fallback":
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🌐 Sin coincidencia local confiable. Conmutando a fallback de internet...',
                'status': 'searching'
            }
            state = await self._nodo_fallback_internet(state)
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🛑 Generación detenida (insegura)...',
                'status': 'searching'
            }
            state = await self._nodo_reset(state)

        # 8. Finalizar
        state = await self._nodo_finalizar(state)

        # Generar respuesta final con Markdown
        final_answer = state["respuesta_final"]
        
        # Si se recuperaron imágenes locales válidas, podemos añadir su enlace al final del Markdown para visualización
        if camino == "generar" and state.get("imagenes_relevantes"):
            final_answer += "\n\n### Micrografías del Manual Relacionadas:\n"
            for i, img in enumerate(state["imagenes_relevantes"]):
                # Servir la imagen mediante la ruta montada
                img_name = os.path.basename(img["path"])
                backend_port = os.getenv("API_PORT", "8000")
                img_url = f"http://127.0.0.1:{backend_port}/histopatologia_data/embeddings/{img_name}"
                desc = img.get("descripcion", f"Figura {i+1}")
                final_answer += f"![{desc}]({img_url})\n\n*{desc}*\n"

        cleanup_memory()

        yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': final_answer,
            'status': 'completed',
            'imagenes_relevantes': state.get("imagenes_relevantes", [])
        }



# ============================================================================
# COMPATIBILIDAD CON API EXISTENTE
# ============================================================================

class AsistenteHistologiaMultimodal(MedicalAgent):
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
        final_state = await self.procesar_consulta_estado(consulta_usuario or "Analizar contenido", imagen_path, imagen_base64)
        return {
            "respuesta": final_state["respuesta_final"],
            "imagenes_relevantes": final_state.get("imagenes_relevantes", []),
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
        res, _ = await self.gestor_qdrant.buscar_muvera_2stage(
            query_mv,
            query_fde,
            top_k=top_k,
            prefetch_multiplier=prefetch_multiplier,
            filtro_tipo="imagen" if image_path else None
        )
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

# Compatibility wrapper for FastAPI server
class HistologyAgent(MedicalAgent):
    """Compatibility class wrapping MedicalAgent as HistologyAgent"""
    pass

