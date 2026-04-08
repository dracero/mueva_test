# Requirements Document

## Introduction

Esta especificación define la optimización del sistema de verificación de imágenes mediante la implementación de verificación condicional de Hamming (dHash). El objetivo es mejorar el rendimiento del sistema evitando verificaciones visuales redundantes cuando la confianza del score de similitud de embeddings es suficientemente alta.

## Glossary

- **Image_Verification_System**: El sistema que verifica coincidencias de imágenes usando embeddings y verificación visual
- **Embedding_Score**: Puntuación de similitud calculada mediante MaxSim directo entre embeddings de imágenes (rango: 0-1000+)
- **Hamming_Verification**: Verificación visual que calcula la distancia de Hamming normalizada entre hashes dHash de imágenes
- **High_Confidence_Threshold**: Umbral de 890 (0.890) por encima del cual la verificación de Hamming no es necesaria
- **Verification_Threshold**: Umbral mínimo de 830 (0.830) para considerar que dos imágenes coinciden semánticamente
- **dHash**: Difference Hash, algoritmo de hashing perceptual usado para comparación visual de imágenes

## Requirements

### Requirement 1: Verificación Condicional de Hamming

**User Story:** Como desarrollador del sistema, quiero que la verificación de Hamming se ejecute solo cuando el score de embeddings esté por debajo de 890, para mejorar el rendimiento sin sacrificar precisión.

#### Acceptance Criteria

1. WHEN THE Embedding_Score is greater than or equal to 890, THE Image_Verification_System SHALL accept the match without executing Hamming_Verification
2. WHEN THE Embedding_Score is less than 890 AND greater than or equal to THE Verification_Threshold, THE Image_Verification_System SHALL execute Hamming_Verification
3. WHEN THE Embedding_Score is less than THE Verification_Threshold, THE Image_Verification_System SHALL reject the match without executing Hamming_Verification
4. FOR ALL accepted matches with Embedding_Score >= 890, THE Image_Verification_System SHALL log that Hamming_Verification was skipped due to high confidence

### Requirement 2: Preservar Comportamiento de Verificación Existente

**User Story:** Como desarrollador del sistema, quiero que el comportamiento de verificación de Hamming se preserve para scores entre 830-889, para mantener la precisión en casos de confianza media.

#### Acceptance Criteria

1. WHEN Hamming_Verification is executed, THE Image_Verification_System SHALL calculate dHash similarity between query and match images
2. WHEN dHash similarity is less than 0.70, THE Image_Verification_System SHALL reject the match as a false positive
3. WHEN dHash similarity is greater than or equal to 0.70, THE Image_Verification_System SHALL accept the match
4. THE Image_Verification_System SHALL log the dHash similarity score for all executed Hamming_Verification operations

### Requirement 3: Logging y Observabilidad

**User Story:** Como desarrollador del sistema, quiero logs claros que indiquen cuándo se ejecuta o se omite la verificación de Hamming, para facilitar el debugging y monitoreo del sistema.

#### Acceptance Criteria

1. WHEN THE Embedding_Score is >= 890, THE Image_Verification_System SHALL log "Match confirmado con alta confianza (score >= 890), verificación visual omitida"
2. WHEN THE Embedding_Score is < 890 AND Hamming_Verification is executed, THE Image_Verification_System SHALL log "Ejecutando Verificación Visual estricta"
3. THE Image_Verification_System SHALL include the Embedding_Score value in all verification log messages
4. THE Image_Verification_System SHALL maintain the existing log format for dHash similarity results

### Requirement 4: Compatibilidad con Configuración Existente

**User Story:** Como administrador del sistema, quiero que el umbral de verificación (830) se mantenga configurable mediante variable de entorno, para permitir ajustes sin cambios de código.

#### Acceptance Criteria

1. THE Image_Verification_System SHALL read the Verification_Threshold from the VERIFICATION_THRESHOLD environment variable
2. WHEN VERIFICATION_THRESHOLD is not set, THE Image_Verification_System SHALL use 830 as the default value
3. THE High_Confidence_Threshold (890) SHALL be defined as a constant in the code
4. THE Image_Verification_System SHALL use both thresholds consistently throughout the verification logic
