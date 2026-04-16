# Implementation Plan: Conversational Text Responses

## Overview

Este plan implementa la funcionalidad de respuestas conversacionales y soporte para consultas de solo texto en el sistema RAG de histopatología. La implementación se centra en refactorizar el método `_nodo_generar_respuesta` para detectar el tipo de consulta y adaptar dinámicamente los prompts y la construcción de mensajes.

## Tasks

- [x] 1. Refactorizar método _nodo_generar_respuesta para separar responsabilidades
  - [x] 1.1 Extraer lógica de detección de tipo de consulta a método _detectar_tipo_consulta
    - Crear método que verifique state['imagen_consulta'] y state['imagenes_relevantes']
    - Retornar 'imagen' si hay contexto visual, 'texto' en caso contrario
    - Validar que imagen_consulta sea un path existente antes de considerarla
    - _Requirements: 3.3, 1.1_
  
  - [x] 1.2 Escribir unit tests para _detectar_tipo_consulta
    - Test: detectar 'imagen' cuando hay imagen_consulta válida
    - Test: detectar 'imagen' cuando hay imagenes_relevantes
    - Test: detectar 'texto' cuando no hay imágenes
    - Test: detectar 'texto' cuando imagen_consulta no existe en filesystem
    - _Requirements: 3.3_

- [x] 2. Implementar generación de prompts adaptativos
  - [x] 2.1 Crear método _generar_prompt_sistema con lógica de selección
    - Implementar prompt conversacional para consultas de texto (sin menciones a imágenes)
    - Implementar prompt conversacional para consultas con imagen (con instrucciones visuales)
    - Ambos prompts deben usar tono amigable y educativo
    - Incluir mensaje de error amigable en ambos prompts para contexto insuficiente
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.4, 5.2, 5.3_
  
  - [x] 2.2 Escribir unit tests para _generar_prompt_sistema
    - Test: prompt de texto contiene elementos conversacionales y no menciona imágenes
    - Test: prompt de imagen contiene instrucciones visuales y tono conversacional
    - Test: ambos prompts mantienen rigor científico (mencionan precisión/evidencia)
    - _Requirements: 2.1, 2.3, 2.4, 3.1, 3.2_

- [ ] 3. Implementar construcción adaptativa de mensajes
  - [x] 3.1 Crear método _construir_mensaje_usuario que adapte contenido según tipo
    - Para consultas de texto: construir mensaje solo con texto y contexto
    - Para consultas con imagen: construir mensaje multimodal con imágenes
    - Incluir historial de conversación si existe en state['contexto_memoria']
    - Respetar límites de imágenes (4 con query, 5 sin query)
    - _Requirements: 1.1, 1.2, 3.1, 3.2, 4.1, 4.2, 4.3, 6.2, 6.3_
  
  - [x] 3.2 Escribir unit tests para _construir_mensaje_usuario
    - Test: mensaje de texto solo contiene partes de tipo 'text'
    - Test: mensaje de imagen incluye partes de tipo 'image_url'
    - Test: respeta límite máximo de imágenes
    - Test: incluye contexto_documentos en el mensaje
    - _Requirements: 1.2, 4.1, 4.2, 6.3_

- [ ] 4. Integrar componentes en _nodo_generar_respuesta
  - [x] 4.1 Modificar _nodo_generar_respuesta para usar nuevos métodos
    - Llamar a _detectar_tipo_consulta al inicio
    - Usar _generar_prompt_sistema con el tipo detectado
    - Usar _construir_mensaje_usuario para ensamblar el mensaje
    - Mantener lógica de invocación del LLM y manejo de respuesta
    - Preservar logging existente y agregar log del tipo de consulta detectado
    - _Requirements: 1.1, 1.3, 3.1, 3.2, 3.3, 4.4, 6.1_
  
  - [x] 4.2 Escribir integration tests con mock del LLM
    - Test: flujo completo para consulta de solo texto
    - Test: flujo completo para consulta con imagen
    - Test: manejo de contexto vacío con mensaje amigable
    - _Requirements: 1.1, 1.3, 5.1, 5.3, 6.1, 6.4_

- [x] 5. Checkpoint - Verificar funcionalidad básica
  - Ejecutar tests unitarios y de integración
  - Verificar que consultas de texto funcionan sin imágenes
  - Verificar que consultas con imagen mantienen funcionalidad existente
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implementar manejo de casos edge y errores
  - [x] 6.1 Agregar validación de contexto insuficiente
    - Verificar longitud mínima de contexto_documentos (>50 caracteres)
    - Si contexto es insuficiente, el prompt debe guiar al LLM a responder con mensaje amigable
    - _Requirements: 5.1, 5.3_
  
  - [x] 6.2 Agregar manejo robusto de errores al cargar imágenes
    - Capturar excepciones al leer archivos de imagen
    - Continuar sin esa imagen si falla (log warning)
    - Si todas las imágenes fallan, tratar como consulta de texto
    - _Requirements: 6.2_
  
  - [x] 6.3 Escribir tests para casos edge
    - Test: contexto vacío o muy corto
    - Test: error al cargar imagen (archivo no existe)
    - Test: todas las imágenes fallan al cargar
    - _Requirements: 5.1, 6.2_

- [x] 7. Final checkpoint - Validación completa
  - Ejecutar suite completa de tests
  - Realizar pruebas manuales con consultas de texto variadas
  - Realizar pruebas manuales con consultas con imagen
  - Verificar tono conversacional en respuestas generadas
  - Verificar que mensajes de error son amigables
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- La implementación mantiene compatibilidad completa con funcionalidad existente
- El diseño no requiere cambios al AgentState ni a la arquitectura del grafo
- Los prompts conversacionales mantienen rigor científico mientras usan tono amigable
