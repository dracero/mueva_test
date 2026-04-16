# Requirements Document

## Introduction

Este documento define los requisitos para mejorar el sistema de histopatología para que sea más conversacional y pueda responder consultas de solo texto sin requerir imágenes. El sistema actual está muy enfocado en búsqueda de imágenes y usa un tono técnico y formal. Esta mejora permitirá al sistema funcionar como un asistente educativo más amigable que puede responder preguntas generales de histopatología usando contexto textual.

## Glossary

- **Sistema**: El sistema de chat de histopatología basado en RAG (Retrieval-Augmented Generation)
- **Consulta_de_Texto**: Una pregunta del usuario que no incluye una imagen adjunta
- **Consulta_con_Imagen**: Una pregunta del usuario que incluye una imagen adjunta para análisis
- **Contexto_Textual**: Fragmentos de texto recuperados de la base de datos mediante búsqueda semántica
- **Contexto_Visual**: Imágenes recuperadas de la base de datos mediante búsqueda por similitud
- **Tono_Conversacional**: Estilo de comunicación amigable, claro y educativo, similar a un profesor explicando a un estudiante
- **Prompt_del_Sistema**: Las instrucciones que guían el comportamiento del modelo de lenguaje en la generación de respuestas
- **Nodo_Generar_Respuesta**: La función `_nodo_generar_respuesta` en el archivo muvera_test.py (líneas 1359-1480)

## Requirements

### Requirement 1: Soporte para Consultas de Solo Texto

**User Story:** Como usuario del sistema, quiero poder hacer preguntas generales de histopatología sin necesidad de adjuntar imágenes, para que pueda obtener información conceptual y teórica de forma rápida.

#### Acceptance Criteria

1. WHEN una Consulta_de_Texto es recibida (sin imagen adjunta), THE Sistema SHALL generar una respuesta basada únicamente en Contexto_Textual recuperado
2. WHEN una Consulta_de_Texto es procesada, THE Sistema SHALL omitir la carga y procesamiento de imágenes en el Prompt_del_Sistema
3. WHEN no hay Contexto_Visual disponible, THE Sistema SHALL estructurar la respuesta enfocándose en explicaciones conceptuales del Contexto_Textual
4. THE Sistema SHALL mantener la capacidad de procesar Consulta_con_Imagen sin degradación de funcionalidad

### Requirement 2: Tono Conversacional en Respuestas

**User Story:** Como usuario del sistema, quiero recibir respuestas en un tono amigable y educativo, para que la experiencia sea más natural y menos intimidante.

#### Acceptance Criteria

1. THE Sistema SHALL modificar el Prompt_del_Sistema para usar Tono_Conversacional en lugar de tono técnico y formal
2. THE Sistema SHALL instruir al modelo para que responda como un profesor amigable explicando a un estudiante
3. WHEN el Sistema genera una respuesta, THE respuesta SHALL evitar lenguaje excesivamente técnico o formal sin perder precisión científica
4. THE Sistema SHALL mantener la precisión y rigor científico mientras usa Tono_Conversacional

### Requirement 3: Adaptación Dinámica del Prompt

**User Story:** Como desarrollador del sistema, quiero que el prompt se adapte según el tipo de consulta, para que las instrucciones sean relevantes al contexto disponible.

#### Acceptance Criteria

1. WHEN una Consulta_de_Texto es procesada, THE Nodo_Generar_Respuesta SHALL generar un Prompt_del_Sistema enfocado en análisis textual
2. WHEN una Consulta_con_Imagen es procesada, THE Nodo_Generar_Respuesta SHALL generar un Prompt_del_Sistema que incluya instrucciones para análisis visual
3. THE Nodo_Generar_Respuesta SHALL detectar la presencia o ausencia de imagen en el estado antes de construir el Prompt_del_Sistema
4. WHEN no hay Contexto_Visual disponible, THE Prompt_del_Sistema SHALL omitir todas las instrucciones relacionadas con análisis de imágenes

### Requirement 4: Estructura de Respuesta Flexible

**User Story:** Como usuario del sistema, quiero que la estructura de la respuesta se adapte al tipo de consulta, para que reciba información organizada de forma relevante.

#### Acceptance Criteria

1. WHEN una Consulta_de_Texto es respondida, THE Sistema SHALL usar una estructura de respuesta enfocada en explicación conceptual
2. WHEN una Consulta_con_Imagen es respondida, THE Sistema SHALL usar una estructura de respuesta que incluya análisis visual
3. THE Sistema SHALL eliminar secciones obligatorias de "Imagen encontrada" y "Análisis Visual" cuando no hay Contexto_Visual
4. THE Sistema SHALL mantener la sección de "Evidencia" citando el Contexto_Textual en ambos tipos de consultas

### Requirement 5: Mensajes de Error Amigables

**User Story:** Como usuario del sistema, quiero recibir mensajes de error claros y amigables cuando no hay información disponible, para que entienda las limitaciones del sistema sin frustración.

#### Acceptance Criteria

1. WHEN no hay contexto relevante recuperado, THE Sistema SHALL responder con un mensaje amigable explicando la limitación
2. THE Sistema SHALL reemplazar el mensaje técnico "No hay suficiente contexto o detalle en las fuentes proporcionadas para dar una respuesta precisa a tu consulta" con un mensaje conversacional
3. WHEN el contexto es insuficiente, THE Sistema SHALL sugerir al usuario reformular la pregunta o proporcionar más detalles
4. THE Sistema SHALL mantener honestidad sobre sus limitaciones usando Tono_Conversacional

### Requirement 6: Preservación de Funcionalidad Existente

**User Story:** Como desarrollador del sistema, quiero que las mejoras no rompan la funcionalidad existente de análisis de imágenes, para que el sistema siga siendo útil para consultas visuales.

#### Acceptance Criteria

1. WHEN una Consulta_con_Imagen es procesada, THE Sistema SHALL mantener todas las capacidades de análisis visual existentes
2. THE Sistema SHALL continuar cargando y procesando imágenes recuperadas cuando estén disponibles
3. THE Sistema SHALL mantener la lógica de límite de imágenes (máximo 4-5 imágenes por restricción de API)
4. THE Sistema SHALL preservar la funcionalidad de comparación entre imagen de consulta e imágenes recuperadas
