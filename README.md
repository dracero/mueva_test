# 🔬 Histopatología RAG Multimodal

**Sistema de Retrieval-Augmented Generation (RAG) multimodal para histopatología** basado en ColPali, MUVERA, LangGraph, Groq Llama-4, Memoria SQLite y una interfaz web conversacional con CopilotKit.

> Este sistema permite consultar un atlas de histopatología mediante texto e imágenes, utilizando inteligencia artificial para extraer ontologías, guardar memoria de la conversación, buscar y generar respuestas fundamentadas en los documentos y figuras indexadas.

---

## 📋 Tabla de Contenidos

1. [Nuevas Features: Respuestas Inteligentes de Texto e Imagen](#-nuevas-features-respuestas-inteligentes-de-texto-e-imagen)
2. [Funcionamiento del Programa y Flujo Completo](#-funcionamiento-del-programa-y-flujo-completo)
3. [Esquema y Arquitectura General](#-esquema-y-arquitectura-general)
4. [Explicación de Dependencias](#-explicación-detallada-de-las-dependencias)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Variables de Entorno (`.env.example`)](#-variables-de-entorno-envexample)
7. [Estructura de Archivos](#-estructura-de-archivos)

---

## 🆕 Nuevas Features: Respuestas Inteligentes de Texto e Imagen

### Respuestas de solo texto por defecto

Cuando el usuario hace preguntas teóricas o conceptuales (ej: "¿Qué es el epitelio?", "Explicame el tejido conectivo"), el sistema responde **únicamente con texto**, sin devolver imágenes. Esto hace que las respuestas sean más rápidas y enfocadas en el contenido conceptual.

### Imágenes solo cuando se solicitan explícitamente

El sistema detecta cuándo el usuario pide ver una imagen mediante palabras clave como "imagen", "foto", "figura", "micrografía", "mostrá", "mostrar", "ver", "visualizar". La detección funciona en tres niveles:

1. **Clasificación LLM**: El modelo Llama-4 analiza la intención de la consulta
2. **Detección determinística**: Fallback por keywords con word boundaries para evitar falsos positivos
3. **Override por upload**: Si el usuario adjunta una imagen, siempre se activa el modo imagen

### Búsqueda semántica de imágenes por texto (MaxSim)

Cuando el usuario solicita una imagen por texto (ej: "Mostrá la imagen de una sarcómera"), el sistema:

1. **Busca texto relevante** en Qdrant para construir el contexto de la respuesta
2. **Obtiene todas las imágenes indexadas** de la base de datos
3. **Genera embeddings del texto asociado** (`contexto_texto`) de cada imagen usando ColPali
4. **Calcula MaxSim** (sum of max similarity per query token) entre la consulta y el texto de cada imagen
5. **Selecciona las top 3 imágenes** con mayor similitud semántica

Este approach es preciso porque compara semánticamente la consulta con la descripción real de cada imagen, no con el embedding visual. Así, "mostrame una sarcómera" encuentra correctamente la imagen cuyo texto dice "Sarcómera. Microfotografía Sarcómera. Tejido: Muscular..." aunque esté en una página diferente al texto teórico.

### Tres rutas de búsqueda claras

El nodo `_nodo_buscar` del grafo LangGraph ahora tiene tres rutas bien separadas:

| Ruta | Condición | Comportamiento |
|---|---|---|
| **Consulta_Texto** | No pide imagen, no adjunta imagen | Solo texto. Excluye todas las imágenes de los resultados |
| **Consulta_Imagen_Texto** | Pide imagen por texto | Busca texto + busca imágenes por MaxSim semántico |
| **Consulta_Imagen_Upload** | Adjunta una imagen | Busca por embedding de imagen con verificación dHash |

### Descripción debajo de cada imagen

Las imágenes recuperadas se muestran en el chat con su **descripción original del manual** (caption + contexto del texto asociado) debajo de cada una. Esto permite al estudiante entender qué muestra cada imagen sin necesidad de leer la respuesta completa.

### Click para agrandar

Cada imagen en el chat es clickeable. Al hacer click se abre un **modal de zoom** que muestra la imagen en tamaño completo con su descripción, permitiendo estudiar los detalles histológicos.

### Manejo de "imágenes no encontradas"

Cuando el usuario pide una imagen pero no se encuentra ninguna relevante en la base de datos, el sistema:
- Informa amablemente que no se encontraron imágenes
- Ofrece una respuesta textual alternativa basada en el contexto disponible
- No muestra galería de imágenes vacía

### Contrato API estable

El endpoint `/copilotkit/chat` ahora:
- Usa `procesar_consulta_estado` directamente para acceder al estado completo del agente
- Garantiza que `imagenes_recuperadas` siempre es una lista (nunca `null`)
- Cada imagen es un objeto `{path, descripcion}` en vez de solo un string

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
3. **Clasificación de intención**: El sistema determina si el usuario quiere solo texto o también imágenes, usando LLM + detección determinística de keywords.
4. **Búsqueda Vectorial a dos Etapas**:
    * **Stage 1 (FDE):** Busca a altísima velocidad en Qdrant entre todos los datos FDE a los `k` resultados posibles.
    * **Stage 2 (Reranking Multi-vector):** Compara los resultados previamente filtrados por FDE con la representación fina Multi-vector de ColPali. Devuelve al RAG exactamente la página del PDF o la Imagen más precisa.
5. **Búsqueda semántica de imágenes** (solo si el usuario pidió imágenes): Calcula MaxSim entre la consulta y el texto asociado a cada imagen para encontrar la más relevante.
6. **Respuestas con Memoria LINEAL**: La respuesta final se forma y se presenta al Frontend. Adicionalmente, usando `SQLite`, la interacción anterior se guarda (pregunta/resumen semántico), así el chatbot contextual no pierde el hilo de la charla.

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
│  │ 1. Recibir │──▶│ 2. Analiza │──▶│ 3. Clasifica│──▶│ 4. Busca  │  │
│  │ img / txt  │   │ Ontología  │   │ intención   │   │ Vectorial │  │
│  └────────────┘   └────────────┘   └─────────────┘   └─────┬─────┘  │
│         ▲                                                  │        │
│    ┌────┴────┐                                     ┌───────▼──────┐ │
│    │ Memoria │◀────────── ( 5. Respuesta Final )  ─┤ 2-STAGE RAG  │ │
│    │ SQLite  │          con Groq (Llama-4-17b)     │ + MaxSim IMG │ │
│    └─────────┘                                     │ QDRANT CLOUD │ │
│                                                    └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Flujo de Decisión de Imágenes

```
Consulta del usuario
    │
    ├── ¿Imagen adjunta? ──Sí──▶ Buscar por embedding de imagen (Path 3)
    │
    └── No
         │
         ├── ¿Pide imagen? ──Sí──▶ Buscar texto + MaxSim imágenes (Path 2)
         │
         └── No ──▶ Solo texto, sin imágenes (Path 1)
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
Deberás crear una carpeta llamada `pdfs` y dejar los libros o atlas de histología objetivo adentro.
Luego ejecuta el script para indexarlos todos:
```bash
uv run python init_db.py --clean
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

### 4. Usar el sistema
- **Preguntas de texto**: Escribí tu pregunta normalmente. Ej: "¿Qué es el epitelio cilíndrico?"
- **Pedir imágenes**: Usá palabras como "mostrá", "imagen", "foto". Ej: "Mostrá la imagen de una sarcómera"
- **Subir imagen**: Usá el botón "Seleccionar Imagen" en la barra lateral para analizar un corte histológico

---

## 🔑 Variables de Entorno (`.env.example`)
La carpeta raíz del sistema debe poseer un archivo llamado `.env` que contenga las API Keys necesarias para que los servicios de Búsqueda Vectorial Cloud y Groq (LLM para responder) funcionen.

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

# Umbrales de búsqueda (ajustar según GPU)
SEARCH_SCORE_THRESHOLD=830
VERIFICATION_THRESHOLD=830
```

---

## 📁 Estructura de Archivos
- **`muvera_test.py`**: El motor del pipeline multimodal, los embebidos ColPali y la estructura del Grafo de Estado de LangGraph con MUVERA. Contiene las funciones puras de clasificación (`detectar_intencion_imagen`, `filtrar_resultados_busqueda`, `extraer_paginas_de_resultados`) y la capa gestora de `SQLite`.
- **`api.py`**: Puentes con el mundo HTTP. Gestiona `/copilotkit/chat` con contrato JSON estable (`{response, imagenes_recuperadas}`).
- **`init_db.py`**: Herramienta de carga e indexación fría. Crea colecciones con índices para `tipo` y `numero_pagina`.
- **`histopatologia_data/`**: Contiene la base de los documentos (Ontología JSON extraída, cache temporal, imágenes extraídas en `embeddings/`).
- **`pyproject.toml`**: Gestión minuciosa de lock-dependencies de `uv`.
- **`frontend/`**: Astro/React, donde el componente de chat `Chat.tsx` conversa directamente por POST al Python API. Incluye galería de imágenes con descripción y modal de zoom.
