# Implementation Plan: Conditional Hamming Verification

## Overview

Este plan implementa la optimización de verificación condicional de Hamming en el sistema de verificación de imágenes. La implementación introduce una lógica de tres niveles basada en el score de embeddings para decidir cuándo ejecutar la verificación visual (dHash), reduciendo cómputo innecesario sin sacrificar precisión.

## Tasks

- [x] 1. Agregar constante HIGH_CONFIDENCE_THRESHOLD y validación de configuración
  - Agregar la constante `HIGH_CONFIDENCE_THRESHOLD = 890.0` cerca de la definición de `UMBRAL_VERIFICACION` en `muvera_test.py`
  - Agregar try-except alrededor de la conversión de `VERIFICATION_THRESHOLD` para manejar valores inválidos
  - _Requirements: 4.1, 4.2, 4.3_

- [ ]* 1.1 Escribir unit tests para configuración de umbrales
  - Verificar que `HIGH_CONFIDENCE_THRESHOLD` tiene el valor correcto (890.0)
  - Verificar que `UMBRAL_VERIFICACION` usa el default 830 cuando la variable de entorno no está definida
  - Verificar que valores inválidos en `VERIFICATION_THRESHOLD` usan el default 830
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 2. Implementar lógica condicional de tres niveles en _nodo_buscar
  - [x] 2.1 Modificar el bloque de verificación en `_nodo_buscar` (líneas ~1260-1280)
    - Reemplazar el if-else actual con la estructura if-elif-else de tres niveles
    - Nivel 1: `if maxsim_directo < UMBRAL_VERIFICACION` → rechazar sin dHash
    - Nivel 2: `elif maxsim_directo >= HIGH_CONFIDENCE_THRESHOLD` → aceptar sin dHash
    - Nivel 3: `else` (830 <= score < 890) → ejecutar dHash
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 Actualizar mensajes de logging para cada nivel de decisión
    - Nivel 1: Mantener mensaje existente "❌ Tejido NO coincide semánticamente → score bajo"
    - Nivel 2: Agregar mensaje "✅ Match confirmado con alta confianza (score >= 890), verificación visual omitida"
    - Nivel 3: Mantener mensaje existente "✅ Tejido coincide semánticamente. Ejecutando Verificación Visual estricta..."
    - Asegurar que todos los mensajes incluyen el valor del score
    - _Requirements: 1.4, 3.1, 3.2, 3.3, 3.4_

  - [ ]* 2.3 Escribir property test para lógica de decisión basada en score
    - **Property 1: Score-based Decision Logic**
    - **Validates: Requirements 1.1, 1.2, 1.3**
    - Usar Hypothesis para generar scores en rango 0-2000
    - Verificar que score < 830 → REJECT sin dHash
    - Verificar que score >= 890 → ACCEPT sin dHash
    - Verificar que 830 <= score < 890 → VERIFY con dHash

  - [ ]* 2.4 Escribir unit tests para casos límite de umbrales
    - Score exactamente en 830 → debe ejecutar dHash
    - Score exactamente en 890 → debe omitir dHash
    - Score en 889.99 → debe ejecutar dHash
    - Score en 890.01 → debe omitir dHash
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 3. Checkpoint - Verificar lógica de decisión
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Preservar comportamiento de verificación dHash existente
  - [x] 4.1 Verificar que el bloque de verificación dHash permanece sin cambios
    - Confirmar que `_verificar_match_visual` se llama solo en el nivel 3 (830-889)
    - Confirmar que el umbral dHash de 0.70 permanece sin cambios
    - Confirmar que los mensajes de logging de dHash permanecen sin cambios
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 4.2 Escribir property test para comportamiento de verificación dHash
    - **Property 3: dHash Verification Behavior**
    - **Validates: Requirements 2.1, 2.2, 2.3**
    - Usar Hypothesis para generar valores de similitud dHash en rango 0.0-1.0
    - Verificar que similitud < 0.70 → REJECT
    - Verificar que similitud >= 0.70 → ACCEPT

  - [ ]* 4.3 Escribir property test para logging de similitud dHash
    - **Property 4: dHash Similarity Logging**
    - **Validates: Requirements 2.4, 3.3**
    - Verificar que cada ejecución de dHash registra el valor de similitud en los logs
    - Usar Hypothesis para generar scores en rango 830-889.99 y similitudes dHash 0.0-1.0

- [x] 5. Implementar logging y observabilidad
  - [x] 5.1 Verificar formato de mensajes de log
    - Confirmar que el mensaje de alta confianza incluye el umbral (890)
    - Confirmar que todos los mensajes incluyen el valor del score
    - Confirmar que el formato de dHash permanece: "Similitud visual (dHash): X.XXXX (umbral: 0.70)"
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 5.2 Escribir property test para logging de alta confianza
    - **Property 2: High Confidence Logging**
    - **Validates: Requirements 1.4, 3.1, 3.3**
    - Usar Hypothesis para generar scores >= 890
    - Verificar que el log contiene "alta confianza" y el valor del score

  - [ ]* 5.3 Escribir property test para inclusión de score en logs
    - **Property 5: Score Value in Verification Logs**
    - **Validates: Requirements 3.3**
    - Usar Hypothesis para generar scores en rango 0-2000
    - Verificar que el score aparece en el output de log para cualquier decisión

  - [ ]* 5.4 Escribir unit tests para formato de mensajes de log
    - Verificar formato exacto del mensaje de alta confianza
    - Verificar formato exacto del mensaje de confianza media
    - Verificar formato exacto del mensaje de dHash
    - _Requirements: 3.1, 3.2, 3.4_

- [x] 6. Checkpoint final - Verificar integración completa
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis library with minimum 100 iterations
- The implementation modifies only the `_nodo_buscar` method in `muvera_test.py`
- Methods `_verificar_match_visual` and `_calcular_dhash` remain unchanged
- Manual testing with real histopathology images is recommended after implementation
