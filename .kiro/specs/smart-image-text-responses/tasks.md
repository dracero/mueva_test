# Plan de Implementación: Respuestas Inteligentes de Texto e Imagen

## Resumen

Este plan convierte el diseño en tareas incrementales de código para garantizar que el sistema RAG responda con solo texto por defecto y entregue imágenes únicamente cuando el usuario las solicita explícitamente. Se refactorizan los nodos existentes del grafo LangGraph (`_nodo_clasificar`, `_nodo_buscar`, `_nodo_generar_respuesta`) y se ajustan `api.py` y `Chat.tsx` para mantener consistencia end-to-end.

## Tareas

- [x] 1. Extraer funciones puras de clasificación y filtrado en `muvera_test.py`
  - [x] 1.1 Crear función pura `detectar_intencion_imagen(texto: str) -> bool` que reciba el texto de la consulta y retorne `True` si contiene al menos una palabra clave de imagen (`imagen`, `foto`, `figura`, `micrografía`, `mostrá`, `mostrar`, `ver`, `visualizar`, `enseñar`, `muéstrame`), `False` en caso contrario. Usar matching case-insensitive. Ubicar la función como método estático o función de módulo en `muvera_test.py`.
    - _Requisitos: 1.1, 1.2, 1.4_

  - [x] 1.2 Crear función pura `filtrar_resultados_busqueda(resultados: List[Dict], requiere_imagen: bool, tiene_imagen_adjunta: bool) -> Tuple[List[Dict], List[str]]` que filtre los resultados de búsqueda según el tipo de consulta. Cuando `requiere_imagen=False` y `tiene_imagen_adjunta=False`, excluir todos los resultados de tipo "imagen" y retornar `imagenes_relevantes=[]`. Cuando se incluyen imágenes, limitar a máximo 3 resultados de tipo "imagen". Retornar la tupla `(resultados_filtrados, imagenes_relevantes)`.
    - _Requisitos: 2.1, 3.6, 6.1_

  - [x] 1.3 Crear función pura `extraer_paginas_de_resultados(resultados: List[Dict]) -> List[int]` que reciba resultados de búsqueda de tipo "texto" y retorne la lista de números de página únicos sin duplicados, preservando el orden de aparición.
    - _Requisitos: 3.2_

  - [x] 1.4 Crear función pura `rerank_imagenes_por_caption(query_embedding: np.ndarray, candidatas: List[Dict], umbral: float = 0.45) -> List[Dict]` que calcule la similitud coseno entre el embedding medio de la consulta y el embedding medio del caption de cada imagen candidata, filtre por umbral ≥ 0.45, y retorne las candidatas ordenadas por similitud descendente.
    - _Requisitos: 3.4, 3.5_

- [ ] 2. Escribir tests de propiedades para las funciones puras extraídas
  - [ ]* 2.1 Escribir test de propiedad para `detectar_intencion_imagen` — textos sin keywords
    - **Propiedad 1: Consultas sin palabras clave de imagen se clasifican como solo texto**
    - Usar Hypothesis para generar cadenas de texto aleatorias que no contengan ninguna keyword de imagen. Verificar que `detectar_intencion_imagen(texto)` retorna `False`.
    - **Valida: Requisitos 1.1**

  - [ ]* 2.2 Escribir test de propiedad para `detectar_intencion_imagen` — textos con keywords
    - **Propiedad 2: Consultas con palabras clave de imagen se clasifican como solicitud de imagen**
    - Usar Hypothesis para generar cadenas de texto aleatorias e insertar al menos una keyword de imagen. Verificar que `detectar_intencion_imagen(texto)` retorna `True`.
    - **Valida: Requisitos 1.2**

  - [ ]* 2.3 Escribir test de propiedad para clasificación con imagen adjunta
    - **Propiedad 3: Imagen adjunta siempre activa modo imagen**
    - Usar Hypothesis para generar textos aleatorios (con y sin keywords) y una ruta de imagen válida (crear archivo temporal). Verificar que el clasificador establece `requiere_imagen=True`.
    - **Valida: Requisitos 1.3**

  - [ ]* 2.4 Escribir test de propiedad para `filtrar_resultados_busqueda` — exclusión de imágenes
    - **Propiedad 4: Filtrado de resultados excluye imágenes en consultas de solo texto**
    - Usar Hypothesis para generar listas aleatorias de resultados mixtos (texto e imagen). Verificar que cuando `requiere_imagen=False` y sin imagen adjunta, ningún resultado tiene `tipo=="imagen"` y `imagenes_relevantes` es `[]`.
    - **Valida: Requisitos 2.1, 6.1**

  - [ ]* 2.5 Escribir test de propiedad para `extraer_paginas_de_resultados`
    - **Propiedad 6: Extracción de páginas de resultados de texto**
    - Usar Hypothesis para generar listas de resultados con números de página aleatorios. Verificar que el resultado contiene exactamente los números de página únicos presentes, sin duplicados.
    - **Valida: Requisitos 3.2**

  - [ ]* 2.6 Escribir test de propiedad para `rerank_imagenes_por_caption`
    - **Propiedad 7: Re-ranking semántico filtra por umbral y ordena por similitud**
    - Usar Hypothesis con `hypothesis.extra.numpy` para generar embeddings aleatorios normalizados. Verificar que solo se incluyen candidatas con similitud ≥ 0.45 y que el resultado está ordenado por similitud descendente.
    - **Valida: Requisitos 3.4, 3.5**

  - [ ]* 2.7 Escribir test de propiedad para límite máximo de imágenes
    - **Propiedad 8: Límite máximo de imágenes en resultados**
    - Usar Hypothesis para generar listas de 0 a 20 resultados de tipo "imagen". Verificar que la salida filtrada contiene como máximo 3 imágenes.
    - **Valida: Requisitos 3.6**

- [x] 3. Checkpoint — Verificar que las funciones puras y sus tests pasan
  - Ejecutar todos los tests con `pytest`. Asegurar que todas las funciones puras están correctamente implementadas y los tests de propiedad pasan. Preguntar al usuario si hay dudas.

- [x] 4. Refactorizar `_nodo_clasificar` en `muvera_test.py`
  - [x] 4.1 Modificar `_nodo_clasificar` para que use `detectar_intencion_imagen()` como fallback determinístico además de la clasificación del LLM. Si `imagen_consulta` existe y es un archivo válido, establecer `requiere_imagen=True` como override sin consultar al LLM para esa decisión. Si el LLM no retorna `REQUIERE_IMAGEN:`, usar `detectar_intencion_imagen(consulta_usuario)` como fallback.
    - _Requisitos: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 4.2 Escribir test de propiedad para `_detectar_tipo_consulta`
    - **Propiedad 5: Detección de tipo de consulta es consistente con el estado**
    - Usar Hypothesis para generar `AgentState` parciales con combinaciones de `imagen_consulta`, `requiere_imagen` e `imagenes_relevantes`. Verificar que retorna `'imagen'` si y solo si hay imagen adjunta válida O (`requiere_imagen=True` Y `imagenes_relevantes` no vacío).
    - **Valida: Requisitos 2.2, 5.1**

- [x] 5. Refactorizar `_nodo_buscar` en `muvera_test.py`
  - [x] 5.1 Reorganizar `_nodo_buscar` para separar claramente las 3 rutas de búsqueda: **Consulta_Texto** (excluir todas las imágenes cuando `requiere_imagen=False` y sin upload), **Consulta_Imagen_Texto** (ejecutar búsqueda semántica texto→páginas→imágenes cuando `requiere_imagen=True` sin upload), y **Consulta_Imagen_Upload** (buscar con embedding de imagen cuando hay upload). Usar las funciones puras `filtrar_resultados_busqueda`, `extraer_paginas_de_resultados` y `rerank_imagenes_por_caption` extraídas en la tarea 1. Garantizar el invariante: cuando `requiere_imagen=False` y sin upload, `imagenes_relevantes` DEBE ser `[]`.
    - _Requisitos: 2.1, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2_

- [x] 6. Refactorizar `_nodo_generar_respuesta` en `muvera_test.py`
  - [x] 6.1 Modificar `_nodo_generar_respuesta` para manejar el caso de "imágenes solicitadas pero no encontradas". Cuando `requiere_imagen=True` pero `imagenes_relevantes` está vacío, generar una respuesta que informe al usuario que no se encontraron imágenes relevantes y ofrezca una respuesta textual alternativa basada en el contexto de texto disponible.
    - _Requisitos: 5.1, 5.2, 5.3_

- [x] 7. Checkpoint — Verificar refactorización del backend
  - Ejecutar todos los tests con `pytest`. Verificar que los nodos refactorizados funcionan correctamente. Preguntar al usuario si hay dudas.

- [x] 8. Ajustar contrato API en `api.py`
  - [x] 8.1 Modificar `chat_endpoint` en `api.py` para garantizar que `imagenes_recuperadas` siempre sea una lista (nunca `None`). Usar `procesar_consulta_estado` en lugar de `iniciar_flujo_multimodal` para acceder directamente al `AgentState` final y extraer `imagenes_relevantes`. Convertir `None` a `[]` antes de enviar la respuesta. Asegurar que el contrato JSON sea `{"response": str, "imagenes_recuperadas": List[str]}`.
    - _Requisitos: 2.3, 4.1, 6.3, 6.4_

- [x] 9. Ajustar renderizado condicional en `Chat.tsx`
  - [x] 9.1 Verificar y reforzar la lógica de renderizado condicional de la galería de imágenes en `Chat.tsx`. La galería debe renderizarse si y solo si `msg.images` existe y `msg.images.length > 0`. La lógica actual ya es correcta (`{msg.images && msg.images.length > 0 && ...}`), pero verificar que no haya otros puntos donde se rendericen imágenes fuera de esta condición.
    - _Requisitos: 2.4, 4.2, 4.3, 6.5_

  - [ ]* 9.2 Escribir test de propiedad para renderizado condicional de galería (frontend)
    - **Propiedad 9: Frontend renderiza galería si y solo si hay imágenes**
    - Instalar `fast-check` y `@testing-library/react` como dependencias de desarrollo en `frontend/package.json`. Crear test que genere mensajes aleatorios con arrays de imágenes de longitud variable (0 a 5). Verificar que la galería se renderiza si y solo si `images.length > 0`.
    - **Valida: Requisitos 6.5**

- [x] 10. Checkpoint final — Verificar integración end-to-end
  - Ejecutar todos los tests de backend con `pytest` y los tests de frontend (si existen). Verificar que el flujo completo funciona: consultas de texto devuelven `imagenes_recuperadas: []`, consultas con keywords de imagen activan la búsqueda semántica, y el frontend renderiza correctamente. Preguntar al usuario si hay dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido.
- Cada tarea referencia requisitos específicos para trazabilidad.
- Los checkpoints aseguran validación incremental.
- Los tests de propiedad validan propiedades universales de correctitud definidas en el diseño.
- Los tests unitarios validan ejemplos específicos y casos borde.
- **Python PBT**: Hypothesis (ya en `pyproject.toml` como dependencia de desarrollo).
- **TypeScript PBT**: fast-check (necesita instalarse en `frontend/package.json`).
