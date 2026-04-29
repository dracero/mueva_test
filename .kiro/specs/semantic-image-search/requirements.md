# Documento de Requisitos — Búsqueda Semántica de Imágenes

## Introducción

Actualmente, cuando un usuario realiza una consulta de texto que requiere imágenes (`requiere_imagen=True`), el sistema sigue un proceso indirecto de múltiples pasos: primero busca chunks de texto semánticamente similares, luego extrae referencias de figuras de esos textos, y finalmente hace scroll de todas las imágenes en Qdrant para emparejar por referencia de figura. Si no encuentra coincidencias, recurre a un fallback por proximidad de página.

Este enfoque es frágil porque depende de que los textos mencionen explícitamente las figuras, y falla cuando las descripciones de las imágenes son semánticamente relevantes pero no están referenciadas en los chunks de texto recuperados.

La mejora propuesta simplifica este flujo: cuando `requiere_imagen=True`, el sistema debe comparar directamente la consulta del usuario contra las descripciones (captions) de **todas** las imágenes almacenadas en Qdrant usando similitud semántica, y devolver únicamente las 3 imágenes con mayor similitud, siempre que superen un umbral mínimo de relevancia.

## Glosario

- **Sistema_RAG**: El sistema RAG multimodal de histología implementado en `SistemaRAGColPaliPuro`, que orquesta la búsqueda y generación de respuestas mediante LangGraph.
- **Nodo_Buscar**: El nodo `_nodo_buscar` del grafo LangGraph que ejecuta la lógica de búsqueda según el tipo de consulta (Path 1, Path 2, Path 3).
- **Path_2**: La rama de ejecución dentro de `Nodo_Buscar` que se activa cuando `requiere_imagen=True` y no hay imagen adjunta. Actualmente implementa búsqueda indirecta por referencias de figuras.
- **Procesador_ColPali**: El componente `ProcesadorColPaliPuro` que genera embeddings multi-vector usando el modelo ColPali v1.2 tanto para texto como para imágenes.
- **Gestor_Qdrant**: El componente `GestorQdrantMuvera` que gestiona las colecciones de vectores en Qdrant y ejecuta búsquedas MUVERA de dos etapas (FDE rápida + reranking MV preciso).
- **Caption**: El campo `texto` del payload de cada punto de tipo `"imagen"` en Qdrant, que contiene la descripción de la imagen (ej: "Imagen 11.5 - Epitelio estratificado plano...").
- **Similitud_Semántica**: La similitud coseno calculada entre el embedding medio de la consulta del usuario y el embedding medio del caption de cada imagen, ambos generados por `Procesador_ColPali`.
- **Umbral_Similitud**: El valor mínimo de similitud coseno que una imagen debe alcanzar para ser considerada relevante. Valor por defecto: 0.45.
- **Rerank_Caption**: La función `rerank_imagenes_por_caption` que calcula la similitud semántica entre la consulta y los captions de las imágenes candidatas, filtra por umbral absoluto y relativo, y ordena por relevancia descendente.

## Requisitos

### Requisito 1: Búsqueda directa de imágenes por similitud semántica de caption

**Historia de Usuario:** Como usuario del sistema de histología, quiero que cuando hago una consulta de texto que requiere imágenes, el sistema busque directamente en las descripciones de todas las imágenes por similitud semántica, para obtener las imágenes más relevantes sin depender de referencias indirectas en el texto.

#### Criterios de Aceptación

1. WHEN `requiere_imagen` es `True` y no hay imagen adjunta, THE Nodo_Buscar SHALL obtener todos los puntos de tipo `"imagen"` de la colección de Qdrant.
2. WHEN los puntos de imagen son obtenidos, THE Nodo_Buscar SHALL generar el embedding del caption de cada imagen usando Procesador_ColPali y construir una lista de candidatas con su `caption_embedding`.
3. WHEN las candidatas están preparadas, THE Nodo_Buscar SHALL invocar Rerank_Caption con el embedding de la consulta optimizada y la lista de candidatas para calcular la Similitud_Semántica de cada una.
4. WHEN Rerank_Caption retorna resultados, THE Nodo_Buscar SHALL seleccionar como máximo las 3 imágenes con mayor Similitud_Semántica que superen el Umbral_Similitud.
5. WHEN las imágenes seleccionadas existen en el sistema de archivos, THE Nodo_Buscar SHALL almacenar cada imagen como un diccionario con claves `path` y `descripcion` en el campo `imagenes_relevantes` del estado del agente.
6. IF ninguna imagen supera el Umbral_Similitud, THEN THE Nodo_Buscar SHALL establecer `imagenes_relevantes` como una lista vacía.

### Requisito 2: Eliminación de la búsqueda indirecta por referencias de figuras

**Historia de Usuario:** Como desarrollador del sistema, quiero que el Path_2 deje de usar la estrategia de búsqueda indirecta (texto → figuras → imágenes → rerank), para simplificar el flujo y hacerlo puramente semántico.

#### Criterios de Aceptación

1. WHEN `requiere_imagen` es `True` y no hay imagen adjunta, THE Nodo_Buscar SHALL omitir la extracción de referencias de figuras de los chunks de texto.
2. WHEN `requiere_imagen` es `True` y no hay imagen adjunta, THE Nodo_Buscar SHALL omitir el scroll de imágenes para emparejar por campo `figuras`.
3. WHEN `requiere_imagen` es `True` y no hay imagen adjunta, THE Nodo_Buscar SHALL omitir el fallback de búsqueda por proximidad de página.
4. THE Nodo_Buscar SHALL mantener la búsqueda de texto semánticamente similar para construir el contexto del LLM, independientemente de la búsqueda de imágenes.

### Requisito 3: Límite estricto de 3 imágenes en resultados

**Historia de Usuario:** Como usuario, quiero recibir como máximo 3 imágenes por consulta, para que la respuesta sea concisa y relevante sin saturar la interfaz.

#### Criterios de Aceptación

1. THE Nodo_Buscar SHALL limitar la cantidad de imágenes en `imagenes_relevantes` a un máximo de 3 elementos.
2. WHEN más de 3 imágenes superan el Umbral_Similitud, THE Nodo_Buscar SHALL seleccionar las 3 con mayor Similitud_Semántica.
3. WHEN menos de 3 imágenes superan el Umbral_Similitud, THE Nodo_Buscar SHALL incluir solo las imágenes que superan el umbral, sin rellenar con imágenes de menor relevancia.

### Requisito 4: Preservación del contexto textual para generación de respuesta

**Historia de Usuario:** Como usuario, quiero que la respuesta del asistente siga basándose en los textos relevantes encontrados, incluso cuando se muestran imágenes, para que la explicación textual sea coherente con las imágenes mostradas.

#### Criterios de Aceptación

1. WHEN `requiere_imagen` es `True`, THE Nodo_Buscar SHALL ejecutar la búsqueda MUVERA de dos etapas para obtener chunks de texto semánticamente similares a la consulta optimizada.
2. WHEN los chunks de texto son recuperados, THE Nodo_Buscar SHALL usar esos resultados como contexto para la generación de respuesta por el LLM.
3. THE Nodo_Buscar SHALL ejecutar la búsqueda de imágenes por caption de forma independiente a la búsqueda de texto, sin que una afecte los resultados de la otra.

### Requisito 5: Reutilización de la función de reranking existente

**Historia de Usuario:** Como desarrollador, quiero que la búsqueda semántica de imágenes reutilice la función `rerank_imagenes_por_caption` ya existente, para mantener consistencia en el cálculo de similitud y evitar duplicación de lógica.

#### Criterios de Aceptación

1. THE Nodo_Buscar SHALL usar la función Rerank_Caption existente para calcular y ordenar la Similitud_Semántica entre la consulta y los captions de las imágenes.
2. THE Rerank_Caption SHALL mantener su comportamiento actual: filtro por umbral absoluto (por defecto 0.45) y filtro relativo (75% del score máximo).
3. WHEN Rerank_Caption recibe candidatas sin `caption_embedding`, THE Rerank_Caption SHALL omitir esas candidatas del ranking sin generar errores.
