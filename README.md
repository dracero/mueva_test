# Histología RAG Multimodal - Local

Sistema RAG multimodal para histopatología con ColPali, ColBERT, MUVERA y Gemini.

## Requisitos

- Python 3.10+
- UV (gestor de paquetes)
- CUDA (opcional, para GPU)

## Configuración

1. **Completar las API Keys en `.env`:**

```bash
# Editar el archivo .env con tus credenciales
GOOGLE_API_KEY="tu_api_key_de_google"
QDRANT_URL="tu_url_de_qdrant"
QDRANT_KEY="tu_api_key_de_qdrant"
LANGSMITH_API_KEY="tu_api_key_de_langsmith"  # Opcional
```

2. **Instalar dependencias con UV:**

```bash
uv sync
```

## Uso

1. **Colocar los archivos PDF** en la carpeta `./pdfs/` o en el directorio raíz del proyecto.

2. **Ejecutar el script:**

```bash
uv run muvera_test.py
```

## Modos de Operación

El sistema ofrece 3 modos:

1. **Consulta RAG Normal** - Hacer preguntas sobre histología
2. **Verificar Descripción de Estudiante** - Evaluar descripciones de imágenes histológicas
3. **Salir**

## Características

- **ColBERT v2.0** para embeddings de texto (late interaction)
- **ColPali v1.2** para embeddings de imágenes histológicas
- **MUVERA** para two-stage retrieval eficiente
- **Gemini 2.5 Flash** para generación de respuestas
- **RAGAS** para evaluación de calidad
- **LangSmith** para telemetría (opcional)

## Solución de Problemas

### Error `ONNXRuntimeError` o `NoSuchFile`

Si interrumpes la descarga de los modelos (por ejemplo con Ctrl+C), es posible que la caché se corrompa. Para solucionarlo, borra el directorio de caché:

```bash
rm -rf /tmp/fastembed_cache
```

Y vuelve a ejecutar el script.
