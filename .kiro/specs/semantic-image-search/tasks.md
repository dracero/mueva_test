# Plan de Implementación: Búsqueda Semántica de Imágenes

## Resumen

Reemplazar la lógica del Path 2 en `_nodo_buscar` (búsqueda indirecta por referencias de figuras con fallback por proximidad de página) por una comparación semántica directa contra los captions de todas las imágenes en Qdrant. Se mantiene la búsqueda de texto MUVERA 2-stage para contexto del LLM como flujo paralelo independiente.

## Tareas

- [x] 1. Reemplazar la lógica del Path 2 en `_nodo_buscar`
  - [x] 1.1 Eliminar la lógica de búsqueda indirecta del Path 2 actual
    - En `mueva_test/muvera_test.py`, dentro del bloque `elif requiere_imagen:` del método `_nodo_buscar`, eliminar:
      - Extracción de `figuras_referenciadas` y `paginas_relevantes` de los chunks de texto
      - Intento A+B: match de imágenes por campo `figuras` del payload
      - Fallback C: búsqueda de imágenes por proximidad de página (`paginas_expandidas`)
      - Toda la lógica condicional entre los tres intentos
    - Mantener intacto: el Paso 1 de búsqueda de texto MUVERA 2-stage con `filtro_tipo="texto"`
    - Mantener intacto: el scroll de imágenes de Qdrant (ya existe, solo se reutiliza)
    - _Requisitos: 2.1, 2.2, 2.3, 2.4_

  - [x] 1.2 Implementar la búsqueda semántica directa por caption
    - Después del scroll de todas las imágenes, iterar sobre `all_image_points` y para cada punto:
      - Obtener el caption del payload (`texto` o `contexto_texto`)
      - Si el caption no está vacío, generar `caption_embedding` con `self.procesador.generar_embedding_texto(caption)`
      - Si el embedding no es `None`, agregar a la lista de candidatas con `id`, `payload` y `caption_embedding`
    - Invocar `rerank_imagenes_por_caption(query_mv, candidatas, umbral=0.45)` con el embedding de la consulta optimizada
    - Seleccionar las top 3 imágenes del resultado del rerank
    - Para cada imagen seleccionada, verificar que `os.path.exists(img_path)` antes de agregarla a `imagenes_encontradas`
    - Construir cada entrada como `{"path": img_path, "descripcion": caption[:300]}`
    - Asignar `state["imagenes_relevantes"] = imagenes_encontradas` (lista vacía si ninguna supera el umbral)
    - Asignar `resultados = resultados_texto` para que el contexto del LLM use solo texto
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1_

  - [x] 1.3 Actualizar los mensajes de logging del Path 2
    - Cambiar el mensaje de encabezado a: `"[Path 2 — Consulta_Imagen_Texto] Búsqueda de texto + imágenes por caption semántico"`
    - Eliminar los prints de figuras referenciadas, match por figuras, y fallback por página
    - Agregar prints para: cantidad de candidatas con caption, resultado del rerank, imágenes seleccionadas
    - _Requisitos: 1.1_

- [x] 2. Checkpoint — Verificar que el Path 2 funciona correctamente
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Tests de propiedades y unitarios
  - [ ]* 3.1 Test de propiedad: Construcción de candidatas preserva imágenes con caption
    - **Propiedad 1: Construcción de candidatas preserva imágenes con caption**
    - Crear archivo `mueva_test/tests/test_semantic_image_search.py`
    - Usar `hypothesis` con `@settings(max_examples=100)`
    - Generar listas de puntos de imagen con payloads aleatorios (con/sin campo `texto`, con/sin campo `contexto_texto`)
    - Extraer la lógica de construcción de candidatas a una función pura auxiliar `construir_candidatas_caption(image_points, generar_embedding_fn)` para facilitar el testing
    - Verificar que la lista de candidatas contiene exactamente los puntos con caption no vacío y que cada candidata tiene `caption_embedding` no nulo
    - **Valida: Requisito 1.2**

  - [ ]* 3.2 Test de propiedad: Invariante de selección top-3 con umbral
    - **Propiedad 2: Invariante de selección top-3 con umbral**
    - Generar listas de candidatas rerankeadas con scores aleatorios (floats entre 0.0 y 1.0)
    - Verificar: (a) máximo 3 imágenes, (b) todas con score ≥ umbral, (c) si hay más de 3 que superan umbral, son las 3 de mayor score, (d) si hay menos de 3, no se rellena
    - **Valida: Requisitos 1.4, 3.1, 3.2, 3.3**

  - [ ]* 3.3 Test de propiedad: Formato de salida de imágenes relevantes
    - **Propiedad 3: Formato de salida de imágenes relevantes**
    - Generar payloads de imagen con captions de longitud variable (incluyendo strings > 300 caracteres)
    - Verificar que cada diccionario de salida tiene exactamente las claves `path` (string no vacío) y `descripcion` (string con longitud ≤ 300)
    - **Valida: Requisito 1.5**

  - [ ]* 3.4 Test de propiedad: Rerank respeta umbral absoluto y relativo
    - **Propiedad 4: Rerank respeta umbral absoluto y relativo**
    - Generar embeddings aleatorios normalizados (query + candidatas) como arrays numpy 2D
    - Verificar que todas las candidatas retornadas por `rerank_imagenes_por_caption` tienen similitud coseno ≥ 0.45 Y ≥ 75% del score máximo
    - **Valida: Requisito 5.2**

  - [ ]* 3.5 Test de propiedad: Rerank omite candidatas sin embedding
    - **Propiedad 5: Rerank omite candidatas sin embedding**
    - Generar listas mixtas de candidatas (algunas con `caption_embedding`, otras sin él o con `None`)
    - Verificar que el resultado solo contiene candidatas que originalmente tenían `caption_embedding` válido, sin errores
    - **Valida: Requisito 5.3**

  - [ ]* 3.6 Tests unitarios para casos edge del Path 2
    - Test: ninguna imagen supera el umbral → `imagenes_relevantes` es lista vacía (Requisito 1.6)
    - Test: scroll de Qdrant retorna lista vacía → `imagenes_relevantes` es lista vacía
    - Test: error en scroll de Qdrant → degradación graceful, `imagenes_relevantes = []`, sin excepción propagada
    - Test: independencia entre búsqueda de texto e imágenes — mockear ambos flujos y verificar que no se afectan mutuamente (Requisitos 4.1, 4.3)
    - _Requisitos: 1.6, 2.1, 2.2, 2.3, 4.1, 4.3_

- [ ] 4. Checkpoint final — Verificar que todos los tests pasan
  - Ensure all tests pass, ask the user if questions arise.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Los tests de propiedades validan propiedades universales de correctitud definidas en el diseño
- Los tests unitarios validan casos edge y ejemplos específicos
- No se modifica: `api.py`, `Chat.tsx`, `rerank_imagenes_por_caption()`, `ProcesadorColPaliPuro`, `GestorQdrantMuvera`
