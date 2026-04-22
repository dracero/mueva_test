# Documento de Requisitos: Respuestas Inteligentes de Texto e Imagen

## Introducción

Este feature mejora el sistema RAG multimodal de histopatología para que responda con **solo texto por defecto** cuando el usuario hace preguntas teóricas, y devuelva **imágenes únicamente cuando el usuario las solicita explícitamente**. Cuando se solicitan imágenes por texto, el sistema utiliza **búsqueda semántica por texto** para encontrar chunks de texto similares y luego devolver las imágenes asociadas a esas páginas.

El sistema actual ya tiene lógica parcial para clasificar consultas (`_nodo_clasificar` con `requiere_imagen`) y para buscar imágenes semánticamente (`_nodo_buscar` con "Búsqueda Semántica de Imágenes por Texto"). Sin embargo, el comportamiento no es consistente: imágenes pueden filtrarse al frontend cuando no fueron solicitadas, y la ruta de búsqueda semántica de imágenes por texto puede no activarse correctamente.

## Glosario

- **Sistema_RAG**: La clase `SistemaRAGColPaliPuro` en `muvera_test.py`, que orquesta el flujo RAG multimodal mediante LangGraph.
- **Clasificador**: El nodo `_nodo_clasificar` del grafo LangGraph, responsable de determinar la intención del usuario mediante un LLM.
- **Buscador**: El nodo `_nodo_buscar` del grafo LangGraph, responsable de ejecutar búsquedas en Qdrant y filtrar resultados.
- **Generador**: El nodo `_nodo_generar_respuesta` del grafo LangGraph, responsable de construir el prompt y generar la respuesta final.
- **API_Backend**: El módulo `api.py` con FastAPI que expone el endpoint `/copilotkit/chat` y devuelve la respuesta y las imágenes al frontend.
- **Frontend_Chat**: El componente React `Chat.tsx` que renderiza mensajes y galerías de imágenes.
- **Consulta_Texto**: Una consulta del usuario que es puramente teórica o conceptual, sin solicitar imágenes (ej: "¿Qué es el epitelio?").
- **Consulta_Imagen_Texto**: Una consulta del usuario que solicita explícitamente ver una imagen usando lenguaje textual (ej: "Mostrá la imagen de epitelio cilíndrico").
- **Consulta_Imagen_Upload**: Una consulta del usuario que incluye una imagen adjunta (upload) para análisis.
- **Búsqueda_Semántica_Texto**: El proceso de encontrar imágenes buscando primero chunks de texto semánticamente similares a la consulta, y luego recuperando las imágenes de las mismas páginas.
- **requiere_imagen**: Campo booleano en el estado del agente (`AgentState`) que indica si el usuario solicitó explícitamente ver una imagen.

## Requisitos

### Requisito 1: Clasificación precisa de intención del usuario

**Historia de Usuario:** Como usuario del sistema de histopatología, quiero que el sistema distinga correctamente entre preguntas teóricas y solicitudes de imágenes, para que reciba el tipo de respuesta adecuado.

#### Criterios de Aceptación

1. WHEN el usuario envía una consulta sin palabras clave de imagen (ej: "¿Qué es el epitelio?", "Explicame el tejido conectivo"), THE Clasificador SHALL establecer `requiere_imagen` en FALSE.
2. WHEN el usuario envía una consulta con solicitud explícita de imagen (ej: "Mostrá la imagen de epitelio cilíndrico", "¿Tenés una foto de tejido conectivo?", "Quiero ver una micrografía de cartílago"), THE Clasificador SHALL establecer `requiere_imagen` en TRUE.
3. WHEN el usuario adjunta una imagen (upload), THE Clasificador SHALL tratar la consulta como Consulta_Imagen_Upload independientemente del texto.
4. THE Clasificador SHALL reconocer palabras clave en español que indican solicitud de imagen, incluyendo: "imagen", "foto", "figura", "micrografía", "mostrá", "mostrar", "ver", "visualizar".

### Requisito 2: Respuestas de solo texto por defecto

**Historia de Usuario:** Como estudiante de histopatología, quiero recibir respuestas de solo texto cuando hago preguntas teóricas, para que la respuesta sea rápida y enfocada en el contenido conceptual.

#### Criterios de Aceptación

1. WHEN `requiere_imagen` es FALSE y no hay imagen adjunta, THE Buscador SHALL excluir todos los resultados de tipo "imagen" de los resultados de búsqueda.
2. WHEN `requiere_imagen` es FALSE y no hay imagen adjunta, THE Generador SHALL utilizar el prompt de solo texto (sin referencias a imágenes).
3. WHEN `requiere_imagen` es FALSE y no hay imagen adjunta, THE API_Backend SHALL devolver una lista vacía en el campo `imagenes_recuperadas` de la respuesta JSON.
4. WHEN `requiere_imagen` es FALSE y no hay imagen adjunta, THE Frontend_Chat SHALL renderizar únicamente el contenido textual del mensaje del asistente, sin galería de imágenes.

### Requisito 3: Búsqueda semántica de imágenes por texto

**Historia de Usuario:** Como estudiante de histopatología, quiero solicitar imágenes describiendo lo que busco con texto, para que el sistema encuentre imágenes relevantes basándose en la similitud semántica con el contenido textual asociado.

#### Criterios de Aceptación

1. WHEN `requiere_imagen` es TRUE y no hay imagen adjunta (Consulta_Imagen_Texto), THE Buscador SHALL ejecutar primero una búsqueda de texto para encontrar chunks semánticamente similares a la consulta.
2. WHEN el Buscador encuentra chunks de texto relevantes, THE Buscador SHALL identificar las páginas de origen de esos chunks.
3. WHEN el Buscador identifica páginas relevantes, THE Buscador SHALL buscar imágenes indexadas en esas páginas en la colección de Qdrant.
4. WHEN el Buscador encuentra imágenes candidatas en las páginas relevantes, THE Buscador SHALL re-rankear las imágenes usando similitud coseno entre el embedding de la consulta y el embedding del caption de cada imagen.
5. WHEN la similitud coseno de una imagen candidata es mayor o igual a 0.45, THE Buscador SHALL incluir esa imagen en los resultados.
6. THE Buscador SHALL limitar a un máximo de 3 imágenes en los resultados de búsqueda semántica por texto.

### Requisito 4: Entrega de imágenes al frontend

**Historia de Usuario:** Como estudiante de histopatología, quiero ver las imágenes recuperadas en la interfaz de chat cuando las solicito, para poder estudiar visualmente los tejidos.

#### Criterios de Aceptación

1. WHEN el Buscador recupera imágenes relevantes (ya sea por Consulta_Imagen_Texto o Consulta_Imagen_Upload), THE API_Backend SHALL incluir las rutas de las imágenes en el campo `imagenes_recuperadas` de la respuesta JSON.
2. WHEN la respuesta JSON contiene `imagenes_recuperadas` con una o más rutas, THE Frontend_Chat SHALL renderizar cada imagen en la galería de imágenes debajo del texto de respuesta.
3. WHEN la respuesta JSON contiene `imagenes_recuperadas` como lista vacía, THE Frontend_Chat SHALL omitir la galería de imágenes y mostrar solo texto.

### Requisito 5: Respuesta textual contextualizada para consultas con imagen

**Historia de Usuario:** Como estudiante de histopatología, quiero que cuando solicito una imagen, la respuesta textual también describa y analice la imagen encontrada, para entender qué estoy viendo.

#### Criterios de Aceptación

1. WHEN `requiere_imagen` es TRUE y el Buscador recupera imágenes, THE Generador SHALL utilizar el prompt de tipo "imagen" que instruye al LLM a describir y analizar las imágenes recuperadas.
2. WHEN `requiere_imagen` es TRUE y el Buscador no recupera imágenes relevantes, THE Generador SHALL informar al usuario que no se encontraron imágenes relevantes para su consulta y ofrecer una respuesta textual alternativa.
3. IF el contexto recuperado es insuficiente (menos de 50 caracteres), THEN THE Generador SHALL responder con un mensaje indicando que no tiene suficiente información y sugerir reformular la pregunta.

### Requisito 6: Consistencia del flujo end-to-end

**Historia de Usuario:** Como desarrollador del sistema, quiero que el flujo completo desde la clasificación hasta la respuesta sea consistente, para que no haya fugas de imágenes en respuestas de solo texto ni ausencia de imágenes cuando se solicitan.

#### Criterios de Aceptación

1. THE Sistema_RAG SHALL garantizar que cuando `requiere_imagen` es FALSE, el campo `imagenes_relevantes` del estado sea una lista vacía al finalizar el nodo Buscador.
2. THE Sistema_RAG SHALL garantizar que cuando `requiere_imagen` es TRUE (Consulta_Imagen_Texto), el nodo Buscador ejecute la ruta de Búsqueda_Semántica_Texto antes de filtrar resultados.
3. WHEN el flujo completo se ejecuta para una Consulta_Texto, THE API_Backend SHALL devolver `imagenes_recuperadas` como lista vacía.
4. WHEN el flujo completo se ejecuta para una Consulta_Imagen_Texto, THE API_Backend SHALL devolver `imagenes_recuperadas` con las imágenes encontradas por Búsqueda_Semántica_Texto (si las hay).
5. THE Frontend_Chat SHALL renderizar imágenes si y solo si `imagenes_recuperadas` contiene al menos un elemento.
