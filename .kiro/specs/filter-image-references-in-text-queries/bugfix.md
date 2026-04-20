# Bugfix Requirements Document

## Introduction

When a user makes a text-only query after a previous query with an image, the system incorrectly includes references to images from the previous turn in the conversation history. This causes the LLM to reference images that are not relevant to the current text-only query, leading to confusing and contextually inappropriate responses.

The bug occurs in the `_construir_mensaje_usuario` method in `muvera_test.py`, where the conversation history (`contexto_memoria`) is included as-is without filtering image references based on the current query type.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `tipo_consulta == 'texto'` AND `contexto_memoria` contains image references from previous turns THEN the system includes those image references in the conversation history passed to the LLM

1.2 WHEN the LLM receives a text-only query with image references in the history THEN the system produces responses that mention or reference images not present in the current context

### Expected Behavior (Correct)

2.1 WHEN `tipo_consulta == 'texto'` AND `contexto_memoria` contains image references from previous turns THEN the system SHALL filter out all image references from the conversation history before passing it to the LLM

2.2 WHEN the LLM receives a text-only query with filtered history THEN the system SHALL produce responses based only on textual context without mentioning images from previous turns

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `tipo_consulta == 'imagen'` AND `contexto_memoria` contains image references from previous turns THEN the system SHALL CONTINUE TO include those image references in the conversation history

3.2 WHEN `contexto_memoria` is empty or None THEN the system SHALL CONTINUE TO construct messages without any conversation history

3.3 WHEN `tipo_consulta == 'texto'` AND `contexto_memoria` contains only text (no image references) THEN the system SHALL CONTINUE TO include the full conversation history as-is
