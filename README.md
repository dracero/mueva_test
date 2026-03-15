# 🔬 Histopatología RAG Multimodal

**Sistema de Retrieval-Augmented Generation (RAG) multimodal para histopatología** basado en ColPali, MUVERA, LangGraph, Groq Llama-4, Memoria SQLite y una interfaz web conversacional con CopilotKit.

> Este sistema permite consultar un atlas de histopatología mediante texto e imágenes, utilizando inteligencia artificial para extraer ontologías, guardar memoria de la conversación, buscar y generar respuestas fundamentadas en los documentos y figuras indexadas.

---

## 📋 Tabla de Contenidos

1. [Funcionamiento del Programa y Flujo Completo](#-funcionamiento-del-programa-y-flujo-completo)
2. [Esquema y Arquitectura General](#-esquema-y-arquitectura-general)
3. [Explicación de Dependencias](#-explicación-detallada-de-las-dependencias)
4. [Instalación y Configuración](#-instalación-y-configuración)
5. [Variables de Entorno (`.env.example`)](#-variables-de-entorno-envexample)
6. [Estructura de Archivos](#-estructura-de-archivos)

---

## ⚙️ Funcionamiento del Programa y Flujo Completo

El corazón del sistema es un pipeline completo que involucra extracción, transformación de la data en embeddings, y un asistente que interactúa contextualmente con una base de datos vectorial mediante una interfaz web.

### 1. Ingesta y Procesamiento de Datos (PDFs)
Al inicio o mediante el endpoint de `/reindex`, el sistema toma los documentos PDF en la carpeta `pdfs/`. 
- Se **extraen imágenes (figuras relevantes)** usando `PyMuPDF` directamente del archivo y también se convierte a formato `.jpg`.
- Se extrae el texto usando `PyPDF2`.
- El modelo **Llama-4 (Groq)** procesa el texto general del documento y genera automáticamente un archivo JSON llamado **Ontología**, que contiene la clasificación de tejidos, patologías y sistemas descritos en el libro. 

### 2. Embeddings con ColPali + Compresión FDE con MUVERA
Tanto el texto como las imágenes extraídas se someten al modelo `ColPali v1.2`. A diferencia de modelos tradicionales, ColPali genera "Múltiples Vectores" (Multi-vectors) que preservan el entendimiento especial de las imágenes y la semántica profunda del texto. 
Como estos `Multi-vectores` son muy pesados para buscar rápidamente, se envían por una segunda herramienta llamada **MUVERA**. MUVERA los comprime en un código de dimensión fija (Fixed Dimensional Encodings - FDE).

Ambos datos (el rápido FDE y el preciso Multi-vector) se almacenan en la base de datos distribuida en la nube: **Qdrant**.

### 3. Las Fases de una Consulta del Usuario
El asistente procesa cada consulta mediante un **Grafo de Estados con LangGraph**:
1. El usuario pregunta algo y (opcionalmente) envía una foto de un corte histológico al Frontend. El backend toma el `texto`, procesa cualquier texto de la `imagen` previa del chat, y busca un contexto de `ontología`.
2. El LLM (Llama-4 en Groq) _reformula y optimiza_ la pregunta (ej., transformando lenguaje burdo a tecnicismos histopatológicos aptos para la búsqueda vectorial).
3. **Búsqueda Vectorial a dos Etapas**:
    * **Stage 1 (FDE):** Busca a altísima velocidad en Qdrant entre todos los datos FDE a los `k` resultados posibles.
    * **Stage 2 (Reranking Multi-vector):** Compara los resultados previamente filtrados por FDE con la representación fina Multi-vector de ColPali. Devuelve al RAG exactamente la página del PDF o la Imagen más precisa.
4. **Respuestas con Memoria LINEAL**: La respuesta final se forma y se presenta al Frontend. Adicionalmente, usando `SQLite`, la interacción anterior se guarda (pregunta/resumen semántico), así el chatbot contextual no pierde el hilo de la charla y evita contaminarse como antes sucedía con memorias vectoriales basadas en ChromaDB que arrastraban ruido.

---

## 🏗️ Esquema y Arquitectura General

El sistema se apoya en tres capas integradas: 

### Esquema Visual del Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USUARIO (Frontend React/Astro)             │
│                         Pregunta: "¿Qué glándula es esta?" + IMG     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
            (API REST FastAPI - Enpoints: /copilotkit/chat)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH (Backend)                           │
│  ┌────────────┐   ┌────────────┐   ┌─────────────┐   ┌───────────┐  │
│  │ 1. Recibir │──▶│ 2. Analiza │──▶│ 3. LLM Muta │──▶│ 4. Busca  │  │
│  │ img / txt  │   │ Ontología  │   │ a busqueda  │   │ Vectorial │  │
│  └────────────┘   └────────────┘   └─────────────┘   └─────┬─────┘  │
│         ▲                                                  │        │
│    ┌────┴────┐                                     ┌───────▼──────┐ │
│    │ Memoria │◀────────── ( 5. Respuesta Final )  ─┤ 2-STAGE RAG  │ │
│    │ SQLite  │          con Groq (Llama-4-17b)     │ QDRANT CLOUD │ │
│    └─────────┘                                     └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Explicación Detallada de las Dependencias

El archivo `pyproject.toml` especifica las dependencias core del entorno `uv`. A continuación el detalle de las más cruciales:

### Orquestación de Agentes y Memoria
- **`langgraph`** (>=0.0.1): Permite definir el pipeline del asistente como un grafo de estado finito. Maneja el estado desde recolección de query hasta reoptimización y respuesta.
- **`langchain` / `langchain-community`**, etc: Proporcionan utilidades para cadenas de texto genéricas, si bien el corazón recae en LangGraph.

### Manipulación de Base Vectorial y Retriever
- **`qdrant-client`** (>=1.7.0): Librería cliente oficial para interactuar asincrónicamente con la bbdd `Qdrant` donde se almacenan todos los recortes del libro.
- **`fastembed`** (>=0.2.5): Usada fuertemente para el paso de codificación FDE con el módulo `Muvera`, necesario para colisionar los Multivectores con los candidatos livianos.

### Machine Learning / Visión Artificial (HuggingFace)
- **`colpali-engine`**: Modelo Fundacional (Vision-Language) derivado de PaliGemma para capturar representaciones muy densas de imágenes (ej. biopsias) sin necesitar de OCR (Optical Character Recognition).
- **`torch` (PyTorch)** y **`accelerate`** / **`bitsandbytes`**: Backbone para ejecutar ColPali localmente. `bitsandbytes` habilita la compresión a 4-bit para reducir el consumo drástico de VRAM de la tarjeta de video (GPU) manteniendo calidad en la extracción de vectores.

### Modelos de Lenguaje Cloud (LLMs)
- **`langchain-groq` / `groq`**: Permite llamar a los modelos ultra veloces de Groq (Específicamente usamos la familia Llama 3 o 4) para crear respuestas complejas, extraer la ontología y resumir el contexto de la base de datos `SQLite`.
- **`google-generativeai`**: (Opcional en la arquitectura, originariamente usado por Gemini, pero dejado como fallback).

### Backend HTTP
- **`fastapi`** / **`uvicorn`** / **`python-multipart`**: Facilitan servir los LLMs al frontend React desde el archivo `api.py`. FastAPI abre el websocket y la API JSON, y `python-multipart` se encarga de subir las fotos de las biopsias a `uploads/`.

### Extra y Procesamiento Documental
- **`sqlite3`** (Built-in pero crucial): Maneja la nueva tabla de contexto lineal de interacciones, para un recuento histórico. Sustituto de ChromaDB.
- **`PyPDF2`** / **`pdf2image`** / **`PyMuPDF` (fitz)**: Usados para parsear, transformar e identificar objetos puros (figuras sin ruido) de los Atlas PDF de base.
- **`python-dotenv`**: Autenticación segura local, cargando claves de `.env`.

---

## 🚀 Instalación y Configuración

### 1. Clonar e Instalar Setup (UV Packet Manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd mueva_test
uv sync
```

### 2. PDFs y Base Vectorial Local
Deberás crear una carpeta llamda `pdfs` y dejar los libros o atlas de histología objetivo adentro.
Luego ejecuta el script para indexarlos todos:
```bash
uv run python init_db.py
```

### 3. Ejecutar los Webservers
En dos terminales distintas ejecuta:
Terminal A (Backend):
```bash
uv run python api.py
```
Terminal B (Frontend):
```bash
cd frontend
npm install
npm run dev
```

---

## 🔑 Variables de Entorno (`.env.example`)
La carpeta raíz del sistema debe poseer un archivo llamado `.env` que contenga las API Keys necesarias para que los servicios de Búsqueda Vectorial Cloud y Groq (LLM para responder) funcionen.
El agente ha creado en este momento el archivo `.env.example` en la ruta `/media/dracero/08c67654-6ed7-4725-b74e-50f29ea60cb21/pythonAI-Others/mueva_test/.env.example`.

Ejemplo del formato de claves:
```env
# Clave principal, gestiona ontologia, resumenes de sqlite y respuestas Langgraph
GROQ_API_KEY="gsk_123472481..." 

# Accesos Qdrant Cluster (Donde van los vectores)
QDRANT_URL="https://tu_endpoint.qdrant.io:6333"
QDRANT_KEY="api_key_de_qdrant..."

# Trazos para desarrollo y métricas
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="xxx"
LANGSMITH_PROJECT="muevera_test"
```

---

## 📁 Estructura de Archivos
- **`muvera_test.py`**: El motor del pipeline multimodal, los embebidos ColPali y la estructura del Grafo de Estado de Langgraph con Muvera. También contiene la capa gestora de `SQLite`.
- **`api.py`**: Puentes con el mundo HTTP. Gestiona `/copilotkit/chat` y envíos imagenísticos multi-part.
- **`init_db.py`**: Herramienta de carga e indexación fría.
- **`histopatologia_data/`**: Contiene la base de los documentos (Ontología JSON extraída, cache temporal y vectores).
- **`pyproject.toml`**: Gestión minuciosa de lock-dependencies de `uv`.
- **`frontend/`**: Astro/React, donde el componente de chat `Chat.tsx` conversa directamente por POST al Python API.
