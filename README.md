# 🔬 Histología RAG Multimodal

**Sistema de Retrieval-Augmented Generation (RAG) multimodal para histopatología** basado en ColPali, MUVERA y LangGraph, con interfaz web conversacional.

> Este sistema permite consultar un atlas de histopatología mediante texto e imágenes, utilizando inteligencia artificial para buscar y generar respuestas fundamentadas en los documentos indexados.

---

## 📋 Tabla de Contenidos

1. [Arquitectura General](#-arquitectura-general)
2. [Grafo de Agentes (LangGraph)](#-grafo-de-agentes-langgraph)
3. [Pipeline de Embeddings: ColPali + MUVERA](#-pipeline-de-embeddings-colpali--muvera)
4. [Base de Datos Vectorial (Qdrant)](#-base-de-datos-vectorial-qdrant)
5. [API Backend (FastAPI)](#-api-backend-fastapi)
6. [Frontend (Astro + React)](#-frontend-astro--react)
7. [Flujo Completo de una Consulta](#-flujo-completo-de-una-consulta)
8. [Estructura de Archivos](#-estructura-de-archivos)
9. [Instalación y Ejecución](#-instalación-y-ejecución)
10. [Variables de Entorno](#-variables-de-entorno)
11. [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 🏗️ Arquitectura General

El sistema sigue una arquitectura de **tres capas** que se comunican mediante HTTP:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USUARIO (Navegador)                         │
│                     http://localhost:4321                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Astro + React)                       │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │index.ast│──▶│  Chat.tsx     │──▶│ fetch() API  │                │
│  └─────────┘   │  (React)     │   │  calls       │                │
│                └──────────────┘   └──────┬───────┘                │
└──────────────────────────────────────────┼─────────────────────────┘
                                           │ HTTP POST
                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI - api.py)                        │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │/copilotkit/│  │/upload-image│  │  /reindex     │  │ /health  │ │
│  │   chat     │  │             │  │              │  │          │ │
│  └─────┬──────┘  └─────────────┘  └──────────────┘  └──────────┘ │
│        │                                                           │
│        ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         SistemaRAGColPaliPuro (muvera_test.py)              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│  │  │ LangGraph    │  │ ColPali +    │  │ Gemini 2.5 Flash │  │   │
│  │  │ (8 nodos)    │  │ MUVERA       │  │ (generación LLM) │  │   │
│  │  └──────┬───────┘  └──────────────┘  └──────────────────┘  │   │
│  └─────────┼───────────────────────────────────────────────────┘   │
└────────────┼───────────────────────────────────────────────────────┘
             │ gRPC / HTTP
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    QDRANT CLOUD (Base Vectorial)                    │
│  ┌──────────────────────┐   ┌───────────────────────────┐          │
│  │ histopatologia_      │   │ histopatologia_           │          │
│  │ content_fde          │   │ content_mv                │          │
│  │ (MUVERA fast search) │   │ (Multi-vector reranking)  │          │
│  └──────────────────────┘   └───────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Grafo de Agentes (LangGraph)

El corazón del sistema es un **grafo secuencial de 8 nodos** implementado con [LangGraph](https://python.langchain.com/docs/langgraph). Cada nodo cumple una función específica dentro del pipeline RAG. El grafo se ejecuta **asincrónicamente** para cada consulta del usuario.

### Diagrama del Grafo

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
              ┌──────────────────────┐
         ①   │  recepcionar_consulta │   Recibe la consulta del usuario
              │                      │   y procesa imagen Base64 si existe
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ②   │     inicializar      │   Carga la ontología y marca
              │                      │   el tiempo de inicio
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ③   │  analizar_ontologia  │   Busca términos relevantes
              │                      │   en la ontología histológica
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ④   │     clasificar       │   Clasifica la consulta usando
              │                      │   Gemini 2.5 Flash (LLM)
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ⑤   │  optimizar_consulta  │   Reformula la consulta para
              │                      │   mejorar la búsqueda RAG
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ⑥   │       buscar         │   Genera embeddings y ejecuta
              │                      │   búsqueda 2-stage MUVERA
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ⑦   │  generar_respuesta   │   Genera la respuesta final
              │                      │   usando Gemini + contexto
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
         ⑧   │      finalizar      │   Registra trayectoria y
              │                      │   cierra el flujo
              └──────────┬───────────┘
                         │
                         ▼
                    ┌─────────┐
                    │   END   │
                    └─────────┘
```

### Descripción Detallada de Cada Nodo

#### ① `recepcionar_consulta`
- **Entrada**: Consulta del usuario (texto) + imagen opcional (Base64)
- **Proceso**: Si hay una imagen en formato Base64, la decodifica y la guarda como archivo `.jpg` en la carpeta `uploads/`
- **Salida**: `consulta_usuario` y `imagen_consulta` (ruta al archivo de imagen, si existe)

#### ② `inicializar`
- **Proceso**: Carga la ontología histológica desde disco (archivo JSON) y establece el timestamp de inicio para medir tiempos
- **Salida**: `ontologia` (diccionario) y `tiempo_inicio`

#### ③ `analizar_ontologia`
- **Proceso**: Busca recursivamente en la ontología histológica los términos que coinciden con la consulta del usuario. Extrae hasta 5 términos relevantes
- **Salida**: `contexto_ontologico` (texto con términos encontrados) y `filtros_ontologia` (lista de filtros)

#### ④ `clasificar`
- **Proceso**: Envía la consulta + contexto ontológico + info de imagen adjunta a **Gemini 2.5 Flash** para que clasifique el tipo de consulta (ej: "identificación de tejido", "técnica de tinción", etc.)
- **LLM**: Gemini 2.5 Flash (temperatura=0)
- **Salida**: `clasificacion` (texto con la clasificación)

#### ⑤ `optimizar_consulta`
- **Proceso**: Reformula la consulta original para optimizar la búsqueda RAG. Usa el LLM para expandir la consulta con sinónimos y terminología técnica
- **LLM**: Gemini 2.5 Flash
- **Salida**: `consulta_optimizada`

#### ⑥ `buscar`
- **Proceso**: Este es el nodo más importante. Ejecuta la **búsqueda 2-stage MUVERA** en Qdrant:
  1. Si hay imagen → genera embedding con **ColPali** (procesamiento visual)
  2. Si solo hay texto → genera embedding con **ColPali** (procesamiento de queries textuales)
  3. Convierte el multi-vector a FDE con **MUVERA**
  4. **Stage 1 (Fast)**: Búsqueda por FDE en la colección `content_fde` → obtiene `top_k × 5` candidatos rápidamente
  5. **Stage 2 (Precise)**: Reranking por multi-vector (MaxSim) en la colección `content_mv` → selecciona los `top_k` mejores resultados
  6. Formatea los resultados con scores, fuentes y texto/imágenes
- **Salida**: `resultados_busqueda`, `contexto_documentos`, `imagenes_relevantes`

#### ⑦ `generar_respuesta`
- **Proceso**: Envía al LLM un prompt con **reglas estrictas** para que responda **exclusivamente** basándose en el contexto recuperado de la base de datos. Nunca debe contradecir el contexto ni usar conocimiento propio para identificar tejidos
- **LLM**: Gemini 2.5 Flash con System Prompt restrictivo
- **Salida**: `respuesta_final`

#### ⑧ `finalizar`
- **Proceso**: Registra el timestamp final en la trayectoria del grafo
- **Salida**: Estado final con toda la trayectoria de nodos

### Estado del Grafo (`AgentState`)

Todos los nodos comparten y modifican un estado tipado:

```python
class AgentState(TypedDict):
    messages: list                    # Historial de mensajes (LangGraph)
    consulta_usuario: str            # Texto original del usuario
    imagen_consulta: Optional[str]   # Ruta a imagen adjunta (si existe)
    imagen_base64: Optional[str]     # Imagen en Base64 (entrada)
    contexto_memoria: str            # Contexto de memoria (reservado)
    ontologia: Dict                  # Ontología histológica cargada
    contexto_ontologico: str         # Términos ontológicos relevantes
    clasificacion: str               # Clasificación de la consulta
    consulta_optimizada: str         # Consulta reformulada para RAG
    filtros_ontologia: List[str]     # Filtros extraídos de ontología
    resultados_busqueda: List[Dict]  # Resultados crudos de Qdrant
    contexto_documentos: str         # Contexto formateado para el LLM
    imagenes_relevantes: List[str]   # Rutas a imágenes recuperadas
    respuesta_final: str             # Texto de respuesta generada
    trayectoria: List[Dict]          # Log de nodos visitados + timestamps
    user_id: str                     # Identificador del usuario
    tiempo_inicio: float             # Timestamp de inicio del flujo
```

---

## 🧠 Pipeline de Embeddings: ColPali + MUVERA

### ¿Qué es ColPali?

[ColPali](https://arxiv.org/abs/2407.01449) es un modelo de embeddings visual basado en **PaliGemma** que trata cada página de un documento como una imagen. Genera **multi-vectores** (late interaction), donde cada "parche" de la imagen tiene su propio vector de 128 dimensiones.

**Ventaja clave**: Un solo modelo para texto E imágenes, eliminando la necesidad de modelos separados.

### ¿Qué es MUVERA?

[MUVERA](https://arxiv.org/abs/2405.19504) (Multi-Vector Retrieval via Fixed Dimensional Encodings) es una técnica que convierte los multi-vectores de ColPali en un **único vector de dimensión fija** (FDE = Fixed Dimensional Encoding) para búsqueda rápida tipo ANN (Approximate Nearest Neighbor).

### Proceso de Two-Stage Retrieval

```
                          CONSULTA                              DOCUMENTO (PDF)
                        ┌──────────┐                          ┌──────────────┐
                        │  Texto   │                          │  Página PDF  │
                        │    o     │                          │  como imagen │
                        │  Imagen  │                          └──────┬───────┘
                        └────┬─────┘                                 │
                             │                                       │
                             ▼                                       ▼
                    ┌────────────────┐                       ┌────────────────┐
                    │   ColPali v1.2 │                       │   ColPali v1.2 │
                    │  (query mode)  │                       │  (image mode)  │
                    └────────┬───────┘                       └────────┬───────┘
                             │                                       │
                             ▼                                       ▼
                    ┌────────────────┐                       ┌────────────────┐
                    │  Multi-vector  │                       │  Multi-vector  │
                    │  N × 128 dims  │                       │  M × 128 dims  │
                    └───┬────────┬───┘                       └───┬────────┬───┘
                        │        │                               │        │
                        ▼        ▼                               ▼        ▼
              ┌──────────┐  ┌────────┐                 ┌──────────┐  ┌────────┐
              │  MUVERA  │  │ Guarda │                 │  MUVERA  │  │ Guarda │
              │ FDE 20480│  │ multi- │                 │ FDE 20480│  │ multi- │
              │  dims    │  │ vector │                 │  dims    │  │ vector │
              └────┬─────┘  └───┬────┘                 └────┬─────┘  └───┬────┘
                   │            │                            │            │
                   ▼            ▼                            ▼            ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │                        QDRANT CLOUD                                │
         │                                                                    │
         │    content_fde (20480-dim)         content_mv (128-dim, MaxSim)    │
         │   ┌─────────────────────┐        ┌─────────────────────────┐      │
         │   │  doc1_fde: [0.1..]  │        │  doc1_mv: [[0.2..],    │      │
         │   │  doc2_fde: [0.3..]  │        │            [0.1..],    │      │
         │   │  doc3_fde: [0.5..]  │        │            ...]        │      │
         │   │  ...                │        │  doc2_mv: [[...], ...] │      │
         │   └─────────────────────┘        └─────────────────────────┘      │
         └─────────────────────────────────────────────────────────────────────┘

                               BÚSQUEDA EN 2 ETAPAS
                               ════════════════════

         STAGE 1 (Rápido):                    STAGE 2 (Preciso):
         Query FDE vs content_fde              Query multi-vector vs content_mv
         ┌────────────────────┐                ┌────────────────────────┐
         │ Cosine similarity  │   ──────▶      │ MaxSim (late interac.) │
         │ top_k × 5 cand.    │   candidatos   │ top_k resultados       │
         └────────────────────┘                └────────────────────────┘
```

### Parámetros MUVERA

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `dim` | 128 | Dimensionalidad de cada vector ColPali |
| `k_sim` | 6 | Clusters = 2⁶ = 64 |
| `dim_proj` | 16 | Dimensión de proyección por cluster |
| `r_reps` | 20 | Repeticiones |
| **FDE total** | **20,480** | 64 × 16 × 20 = 20,480 dimensiones |

---

## 💾 Base de Datos Vectorial (Qdrant)

El sistema utiliza **Qdrant Cloud** con dos colecciones que trabajan en conjunto:

### Colecciones

| Colección | Tipo de Vector | Dimensión | Distancia | Propósito |
|-----------|---------------|-----------|-----------|-----------|
| `histopatologia_content_fde` | Vector único (denso) | 20,480 | Cosine | Búsqueda rápida (Stage 1) |
| `histopatologia_content_mv` | Multi-vector (MaxSim) | 128 por vector | Cosine + MaxSim | Reranking preciso (Stage 2) |

### Estructura del Payload

Cada punto almacenado en Qdrant tiene un payload con la siguiente estructura:

**Para documentos tipo texto:**
```json
{
    "pdf_name": "atlas_histologia.pdf",
    "tipo": "texto",
    "texto": "Las células epiteliales del túbulo proximal presentan..."
}
```

**Para documentos tipo imagen:**
```json
{
    "pdf_name": "atlas_histologia.pdf",
    "tipo": "imagen",
    "imagen_path": "histopatologia_data/embeddings/atlas_page_15.jpg",
    "contexto_texto": "Figura 12. Corte transversal de riñón..."
}
```

### Respuesta de Búsqueda

Cuando se hace una consulta, Qdrant devuelve una lista de resultados con el siguiente formato:

```json
[
    {
        "id": "a3f2d1e4-...",
        "score": 0.8542,
        "payload": {
            "pdf_name": "atlas_histologia.pdf",
            "tipo": "texto",
            "texto": "El tejido hepático se compone de hepatocitos..."
        }
    },
    {
        "id": "b7c8e9f0-...",
        "score": 0.7891,
        "payload": {
            "pdf_name": "atlas_histologia.pdf",
            "tipo": "imagen",
            "imagen_path": "histopatologia_data/embeddings/atlas_page_23.jpg",
            "contexto_texto": "Figura 18. Lobulillo hepático clásico..."
        }
    }
]
```

---

## 🌐 API Backend (FastAPI)

El archivo `api.py` expone una API REST con los siguientes endpoints:

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check del servidor |
| `POST` | `/copilotkit/chat` | Endpoint principal de chat |
| `POST` | `/upload-image` | Subida de imágenes para análisis |
| `POST` | `/reindex` | Re-indexación de PDFs en `./pdfs/` |

### `POST /copilotkit/chat`

**Request:**
```json
{
    "messages": [
        {"role": "user", "content": "¿Qué tejido se observa?"},
        {"role": "assistant", "content": "Se observa..."},
        {"role": "user", "content": "¿Y las células?"}
    ],
    "image_path": null,
    "image_base64": "iVBORw0KGgo..."
}
```

**Response:**
```json
{
    "response": "Según el documento recuperado 'atlas_histologia.pdf'..."
}
```

**Flujo interno:**
1. Extrae el último mensaje del array `messages`
2. Busca imagen: primero en `image_base64`, luego en `image_path`, y finalmente la imagen más reciente en `uploads/`
3. Llama a `asistente.iniciar_flujo_multimodal()` que ejecuta el grafo LangGraph completo
4. Retorna la respuesta generada

### `POST /upload-image`

**Request:** `multipart/form-data` con campo `file`

**Response:**
```json
{
    "filename": "muestra_higado.jpg",
    "path": "uploads/muestra_higado.jpg",
    "status": "success"
}
```

### `POST /reindex`

**Response:**
```json
{
    "status": "success",
    "message": "Procesados 3 archivos"
}
```

### Evento de Startup

Al iniciar el servidor, `api.py` ejecuta automáticamente:
1. Limpia el directorio `uploads/` (evita "memoria" de imágenes anteriores)
2. Inicializa todos los componentes (ColPali, MUVERA, Qdrant, LLM)
3. Verifica si hay PDFs en `./pdfs/` y si la colección Qdrant está vacía
4. Si la colección está vacía → ejecuta indexación automática

---

## 🖥️ Frontend (Astro + React)

### Stack Técnico

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Astro | 5.x | Framework SSG/SSR |
| React | 19.x | Componentes interactivos |
| TypeScript | - | Tipado estático |

### Arquitectura del Frontend

```
frontend/
├── src/
│   ├── pages/
│   │   └── index.astro          ← Página principal (entry point)
│   └── components/
│       └── Chat.tsx             ← Componente React del chat
├── astro.config.mjs             ← Configuración de Astro + React
└── package.json
```

### Componente `Chat.tsx`

El componente principal es un chat interactivo con las siguientes funciones:

| Función | Descripción |
|---------|-------------|
| `handleSend()` | Envía mensaje al backend vía `POST /copilotkit/chat` |
| `handleFileUpload()` | Convierte imagen a Base64 y la adjunta al próximo mensaje |
| `handleReindex()` | Solicita re-indexación de PDFs al backend |

### Flujo de Interacción Frontend ↔ Backend

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chat.tsx (React)                         │
│                                                                 │
│  ┌──────────────┐                                               │
│  │ 📷 Seleccionar│─── FileReader.readAsDataURL() ──▶ imageBase64│
│  │    Imagen     │                                              │
│  └──────────────┘                                               │
│                                                                 │
│  ┌──────────────┐     ┌───────────────────────────────────┐     │
│  │ ✏️ Input texto │────▶│ handleSend()                      │     │
│  └──────────────┘     │                                   │     │
│                       │  fetch("http://127.0.0.1:8000     │     │
│                       │        /copilotkit/chat", {        │     │
│                       │    method: "POST",                │     │
│                       │    body: JSON.stringify({           │     │
│                       │      messages: [...],              │     │
│                       │      image_base64: "..."           │     │
│                       │    })                              │     │
│                       │  })                                │     │
│                       └───────────────┬───────────────────┘     │
│                                       │                         │
│  ┌──────────────┐                     │                         │
│  │ 💬 Messages   │◀── setMessages() ◀─┘  response.json()       │
│  │    Display    │                                              │
│  └──────────────┘                                               │
│                                                                 │
│  ┌──────────────┐     ┌───────────────────────────────────┐     │
│  │ 🔄 Re-indexar │────▶│ fetch("/reindex", {method:"POST"})│     │
│  └──────────────┘     └───────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo Completo de una Consulta

A continuación se muestra el recorrido completo desde que el usuario escribe hasta que recibe la respuesta:

```
USUARIO                   FRONTEND                    BACKEND (API)                LANGGRAPH                      QDRANT
  │                         │                            │                           │                              │
  │  1. Escribe pregunta    │                            │                           │                              │
  │  + adjunta imagen       │                            │                           │                              │
  │ ───────────────────────▶│                            │                           │                              │
  │                         │  2. POST /copilotkit/chat  │                           │                              │
  │                         │  {messages, image_base64}  │                           │                              │
  │                         │ ──────────────────────────▶│                           │                              │
  │                         │                            │  3. Ejecutar grafo        │                              │
  │                         │                            │ ─────────────────────────▶│                              │
  │                         │                            │                           │                              │
  │                         │                            │  ① recepcionar_consulta   │                              │
  │                         │                            │     Decodifica Base64     │                              │
  │                         │                            │     Guarda .jpg           │                              │
  │                         │                            │                           │                              │
  │                         │                            │  ② inicializar            │                              │
  │                         │                            │     Carga ontología       │                              │
  │                         │                            │                           │                              │
  │                         │                            │  ③ analizar_ontologia     │                              │
  │                         │                            │     Busca términos        │                              │
  │                         │                            │                           │                              │
  │                         │                            │  ④ clasificar ──▶ Gemini  │                              │
  │                         │                            │     Clasifica consulta    │                              │
  │                         │                            │                           │                              │
  │                         │                            │  ⑤ optimizar  ──▶ Gemini  │                              │
  │                         │                            │     Reformula consulta    │                              │
  │                         │                            │                           │                              │
  │                         │                            │  ⑥ buscar                 │                              │
  │                         │                            │     ColPali ──▶ embedding │                              │
  │                         │                            │     MUVERA ──▶ FDE        │                              │
  │                         │                            │                           │  Stage 1: FDE search         │
  │                         │                            │                           │ ───────────────────────────▶ │
  │                         │                            │                           │  ◀─── top_k×5 candidatos ── │
  │                         │                            │                           │  Stage 2: MV rerank          │
  │                         │                            │                           │ ───────────────────────────▶ │
  │                         │                            │                           │  ◀─── top_k resultados ──── │
  │                         │                            │                           │                              │
  │                         │                            │  ⑦ generar ──▶ Gemini     │                              │
  │                         │                            │     Respuesta con contexto│                              │
  │                         │                            │                           │                              │
  │                         │                            │  ⑧ finalizar              │                              │
  │                         │                            │                           │                              │
  │                         │                            │ ◀──── respuesta_final ────│                              │
  │                         │  4. {"response": "..."}    │                           │                              │
  │                         │ ◀──────────────────────────│                           │                              │
  │  5. Muestra respuesta   │                            │                           │                              │
  │ ◀───────────────────────│                            │                           │                              │
  │                         │                            │                           │                              │
```

---

## 📁 Estructura de Archivos

```
mueva_test/
├── 📄 api.py                      ← API FastAPI (endpoints REST)
├── 📄 muvera_test.py              ← Lógica principal (clases, grafo, procesadores)
├── 📄 init_db.py                  ← Script de inicialización de Qdrant
├── 📄 debug_retrieval.py          ← Utilidad para debug de búsquedas
├── 📄 pyproject.toml              ← Dependencias Python (uv)
├── 📄 .env                        ← Variables de entorno (API keys)
├── 📄 .gitignore
│
├── 📁 pdfs/                       ← Carpeta donde colocar los PDFs a indexar
│   └── atlas_histologia.pdf
│
├── 📁 uploads/                    ← Imágenes subidas por el usuario (temporal)
│
├── 📁 histopatologia_data/        ← Datos generados por el sistema
│   ├── embeddings/                ← Imágenes de páginas PDF extraídas
│   ├── ontologia/                 ← Ontología histológica (JSON)
│   └── cache/                     ← Cache de resultados intermedios
│
└── 📁 frontend/                   ← Aplicación web (Astro + React)
    ├── 📄 astro.config.mjs
    ├── 📄 package.json
    └── 📁 src/
        ├── 📁 pages/
        │   └── index.astro        ← Página principal
        └── 📁 components/
            └── Chat.tsx           ← Componente del chat
```

---

## 🚀 Instalación y Ejecución

### Requisitos Previos

- **Python** >= 3.10
- **Node.js** >= 18
- **GPU NVIDIA** con CUDA (recomendado, el sistema funciona con CPU pero más lento)
- **Poppler** (para convertir PDFs a imágenes): `sudo apt install poppler-utils`
- **Cuenta Qdrant Cloud** o instancia local de Qdrant

### 1. Clonar e instalar dependencias Python

```bash
# Instalar uv si no lo tenés
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias Python
cd mueva_test
uv sync
```

### 2. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Google Gemini API Key
GOOGLE_API_KEY="tu_api_key_aquí"

# Qdrant Cloud
QDRANT_URL="https://tu-cluster.qdrant.io:6333"
QDRANT_KEY="tu_qdrant_api_key"

# LangSmith (opcional, para telemetría)
LANGSMITH_API_KEY="tu_langsmith_key"
LANGCHAIN_TRACING_V2=false
```

### 3. Colocar los PDFs

```bash
mkdir -p pdfs
cp /ruta/a/tus/pdfs/*.pdf pdfs/
```

### 4. Inicializar la base de datos (primera vez)

```bash
uv run python init_db.py
# O para limpiar y recrear:
uv run python init_db.py --clean
```

### 5. Iniciar el Backend

```bash
uv run python api.py
# El servidor arranca en http://127.0.0.1:8000
```

### 6. Iniciar el Frontend

```bash
cd frontend
npm install
npm run dev
# El frontend arranca en http://localhost:4321
```

### 7. Usar el sistema

1. Abrir http://localhost:4321 en el navegador
2. Escribir una consulta sobre histología
3. Opcionalmente, adjuntar una imagen histológica
4. El sistema buscará en los PDFs indexados y generará una respuesta

---

## 🔑 Variables de Entorno

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `GOOGLE_API_KEY` | ✅ | API Key de Google para Gemini 2.5 Flash |
| `QDRANT_URL` | ✅ | URL del cluster Qdrant Cloud |
| `QDRANT_KEY` | ✅ | API Key de Qdrant Cloud |
| `LANGSMITH_API_KEY` | ❌ | API Key de LangSmith (telemetría) |
| `LANGCHAIN_TRACING_V2` | ❌ | Activar tracing de LangChain (`true`/`false`) |

---

## 🛠️ Tecnologías Utilizadas

### Backend (Python)

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **LangGraph** | >= 0.0.1 | Orquestación del grafo de agentes |
| **LangChain** | >= 0.1.0 | Integración con LLMs |
| **ColPali** (colpali-engine) | >= 0.1.0 | Embeddings multimodales (texto + imágenes) |
| **MUVERA** (fastembed) | >= 0.2.5 | Fixed Dimensional Encodings para retrieval rápido |
| **Qdrant Client** | >= 1.7.0 | Cliente para base de datos vectorial |
| **FastAPI** | >= 0.115 | API REST |
| **Gemini 2.5 Flash** | vía API | LLM para clasificación, optimización y generación |
| **PyTorch** | >= 2.0 | Backend de deep learning (CUDA) |
| **BitsAndBytes** | >= 0.43 | Cuantización 4-bit del modelo ColPali |
| **pdf2image** | >= 1.16 | Conversión de PDF a imágenes |

### Frontend (JavaScript/TypeScript)

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **Astro** | 5.x | Framework web (SSG) |
| **React** | 19.x | Componentes interactivos |
| **TypeScript** | - | Tipado |

### Infraestructura

| Servicio | Uso |
|----------|-----|
| **Qdrant Cloud** | Base de datos vectorial (persistente) |
| **Google AI** (Gemini) | LLM generativo |
| **LangSmith** (opcional) | Observabilidad y tracing |
| **NVIDIA CUDA** | Aceleración GPU para ColPali |

---

## 📝 Notas para Alumnos

### Conceptos Clave

1. **RAG (Retrieval-Augmented Generation)**: En lugar de que el LLM responda de memoria, primero se buscan documentos relevantes en una base de datos y luego se le pasa ese contexto al LLM para que genere una respuesta fundamentada.

2. **Late Interaction / Multi-vector**: ColPali no genera un único vector por documento. Genera **múltiples vectores** (uno por cada "parche" visual de la imagen). Esto permite comparaciones más finas entre consulta y documento.

3. **Two-Stage Retrieval**: La búsqueda en dos etapas balancea velocidad y precisión:
   - **Stage 1**: Búsqueda rápida con vectores comprimidos (FDE) para obtener muchos candidatos
   - **Stage 2**: Reranking preciso con multi-vectores (MaxSim) para seleccionar los mejores

4. **Ontología**: Un vocabulario estructurado del dominio (histopatología) que ayuda al sistema a entender mejor las consultas del usuario.

5. **Query Optimization**: El paso de reformular la consulta antes de buscar mejora significativamente la calidad de los resultados de recuperación.

### ¿Por qué ColPali puro (sin ColBERT)?

- ✅ **Un solo modelo** para texto e imágenes (menos complejidad)
- ✅ **Menos memoria GPU** (~30% reducción)
- ✅ **Consistencia total** en el espacio de embeddings
- ✅ **Código más simple** (~20% menos líneas)
