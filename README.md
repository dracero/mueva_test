# 🔬 MUVERA: RAG Multimodal de Histopatología v1.2

Sistema avanzado de **Generación Aumentada por Recuperación (RAG) Multimodal** especializado en el análisis de imágenes y textos de histopatología. Utiliza una arquitectura de agentes orquestada con **LangGraph**, una base de datos vectorial **Qdrant** y el modelo de vanguardia **ColPali v1.2**.

## 🏗️ Arquitectura del Sistema

El sistema se divide en tres capas principales que garantizan una comunicación fluida y un procesamiento de alta precisión:

1.  **Frontend (React/A2UI)**: Interfaz moderna optimizada para la visualización de muestras histológicas. Permite la carga de imágenes en formato *raw* para evitar la degradación de datos que afectaría la precisión de los embeddings.
2.  **Backend (FastAPI)**: Orquestador que maneja las peticiones REST, la gestión de archivos estáticos y la interfaz con el motor de IA.
3.  **Capa de IA (MUVERA Core)**: Implementación de la lógica de recuperación y generación utilizando LangGraph.

---

## 🤖 Agentes y Flujo (LangGraph)

El "cerebro" del sistema es un grafo de estados que procesa las consultas a través de nodos especializados:

### 1. `_nodo_buscar` (Retrieval)
Este agente es responsable de entender la intención del usuario y recuperar la información más relevante:
-   **Detección de Intento**: Clasifica si la consulta es de texto, de imagen o comparativa.
-   **Recuperación Multimodal**: Realiza búsquedas simultáneas en Qdrant para encontrar chunks de texto y figuras visuales.
-   **Lógica de Filtrado**: Aplica el umbral de verificación (`VERIFICATION_THRESHOLD`) para descartar resultados ruidosos.

### 2. `_nodo_generar_respuesta` (Generation)
Utiliza un modelo **Groq Llama-4 Scout** para sintetizar la respuesta final:
-   **Razonamiento Médico**: Combina el contexto textual del manual con el análisis visual de las imágenes recuperadas.
-   **Referencias Cruzadas**: Identifica estructuras específicas y las vincula con las figuras correspondientes (ej. "En la Imagen 12.1 se observa...").

---

## 📊 Bases de Datos y Vector Stores

### **Qdrant: El Corazón Vectorial**
Utilizamos Qdrant con una configuración de **Arquitectura Dual** para optimizar velocidad y precisión:
-   **Colección FDE (Fixed Dimensional Encoding)**: Una representación comprimida de los embeddings para búsquedas iniciales ultrarrápidas (Stage 1).
-   **Colección Multi-Vector (MaxSim)**: Almacena los embeddings completos de ColPali (128D por patch/token). Se utiliza para el re-ranking de precisión (Stage 2).

### **ColPali v1.2: El Modelo Unificado**
A diferencia de sistemas RAG tradicionales que usan modelos separados para texto e imagen, MUVERA utiliza **ColPali**:
-   Mapea texto e imágenes al mismo espacio vectorial.
-   Permite comparar una consulta textual directamente con "patches" de una imagen microscópica.
-   Especialmente efectivo en documentos visuales complejos como atlas de histología.

---

## 🔍 Lógica de Recuperación de 3 Niveles

Para garantizar que el sistema nunca muestre información incorrecta en un contexto médico, implementamos una verificación de tres niveles:

| Nivel | Score (ColPali) | Acción | Verificación Adicional |
| :--- | :--- | :--- | :--- |
| **Alta Confianza** | `≥ 900` | **Aceptación Directa** | Ninguna (Match semántico fuerte). |
| **Duda Razonable** | `830 - 900` | **Verificación dHash** | Se compara el hash perceptual (dHash) para confirmar identidad visual. |
| **Baja Confianza** | `< 830` | **Rechazo Automático** | El sistema informa que no encontró la imagen específica. |

> [!NOTE]
> El **dHash (Difference Hash)** es crucial porque es robusto a cambios de brillo o compresión, permitiendo confirmar si una imagen recuperada es realmente la que el usuario está consultando, independientemente del score semántico.

---

## 🛠️ Configuración y Desarrollo

### Requisitos Técnicos
-   **GPU**: Recomendado 8GB+ VRAM (probado en RTX 3050 y GTX 1070).
-   **Docker**: Requerido para ejecutar la base de datos vectorial localmente.
-   **Backend**: Python 3.10+ (gestión con `uv`).
-   **Frontend**: Node.js 18+.

### Variables de Entorno Clave (`.env`)
```env
# Configuración de base de datos vectorial Qdrant Local
QDRANT_URL="http://localhost:6333"
QDRANT_KEY=""

# Umbrales y parámetros de búsqueda
SEARCH_SCORE_THRESHOLD=830   # Umbral base de rechazo
VERIFICATION_THRESHOLD=830   # Umbral para disparo de dHash
NORMALIZE_EMBEDDINGS=true    # Consistencia entre diferentes arquitecturas de GPU
QUANTIZATION_BITS=8          # Precisión de los scores (8 o 4 bits)
```

---

## 🚀 Cómo Ejecutar

El proyecto incluye scripts en `package.json` para facilitar la ejecución y el ciclo de desarrollo local:

### 1. Iniciar la Base de Datos (Qdrant en Docker)
```bash
npm run docker:up
```
*Esto iniciará el contenedor local de Qdrant en segundo plano (puertos `6333` y `6334`), persistiendo los datos en `histopatologia_data/qdrant_storage/`.*

### 2. Inicializar / Indexar la Base de Datos
```bash
uv run python init_db.py
```
*Este comando procesará los PDFs de `./pdfs/` y cargará los embeddings en la base de datos local.*

### 3. Ejecutar la Aplicación (Backend + Frontend)
```bash
npm run dev
```
*Este comando iniciará concurrentemente el servidor FastAPI en el puerto `8000` y el cliente frontend de Astro en el puerto `4321`.*

### 4. Limpiar la Base de Datos (Opcional)
Si deseas borrar las colecciones y reiniciar la base de datos de Qdrant de forma limpia:
```bash
npm run db:clear
```

### 5. Apagar la Base de Datos
Para detener y eliminar el contenedor de Qdrant:
```bash
npm run docker:down
```

---
*Desarrollado para el análisis avanzado de muestras histológicas mediante Inteligencia Artificial Agentica.*
