# Filter Image References in Text Queries Bugfix Design

## Overview

This bugfix addresses an issue where text-only queries incorrectly include image references from previous conversation turns in the conversation history. When a user makes a text query after an image query, the system passes the unfiltered conversation history to the LLM, causing it to reference images that are not present in the current context. This leads to confusing responses that mention images the user cannot see.

The fix involves filtering the `contexto_memoria` field to remove image reference markers (`[IMAGEN RECUPERADA]` and `[IMAGEN DE CONSULTA DEL USUARIO]`) when `tipo_consulta == 'texto'`, while preserving the textual content of the conversation history.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when a text-only query includes unfiltered conversation history containing image reference markers from previous turns
- **Property (P)**: The desired behavior - conversation history should be filtered to remove image references when constructing text-only queries
- **Preservation**: Existing behavior for image queries and empty history that must remain unchanged by the fix
- **_construir_mensaje_usuario**: The method in `muvera_test.py` (lines 1456-1600) that constructs the user message content based on query type
- **contexto_memoria**: The conversation history field in AgentState that may contain image reference markers from previous turns
- **tipo_consulta**: The query type parameter that determines whether the query is 'texto' or 'imagen'
- **Image Reference Markers**: Text patterns like `[IMAGEN RECUPERADA N: filename]` and `[IMAGEN DE CONSULTA DEL USUARIO]` that indicate where images were present in previous turns

## Bug Details

### Bug Condition

The bug manifests when a user makes a text-only query (`tipo_consulta == 'texto'`) after a previous query that involved images. The `_construir_mensaje_usuario` method includes the conversation history (`contexto_memoria`) as-is without filtering out image reference markers, causing the LLM to generate responses that reference images not present in the current context.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (AgentState, str)
  OUTPUT: boolean
  
  LET state = input.state
  LET tipo_consulta = input.tipo_consulta
  
  RETURN tipo_consulta == 'texto'
         AND state.get("contexto_memoria") is not None
         AND state["contexto_memoria"] != ""
         AND ('[IMAGEN RECUPERADA' IN state["contexto_memoria"] 
              OR '[IMAGEN DE CONSULTA DEL USUARIO]' IN state["contexto_memoria"])
END FUNCTION
```

### Examples

- **Example 1**: User uploads an image asking "¿Qué estructura es esta?" (Turn 1 - image query). System responds with image analysis. User then asks "¿Cuáles son sus características?" (Turn 2 - text query). The conversation history includes `[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]` markers, causing the LLM to say "Como puedes ver en la imagen..." even though no image is present in Turn 2.

- **Example 2**: User asks "Muéstrame ejemplos de epitelio" with an image (Turn 1). System retrieves and shows 3 images with markers `[IMAGEN RECUPERADA 1]`, `[IMAGEN RECUPERADA 2]`, `[IMAGEN RECUPERADA 3]`. User follows up with text query "¿Qué tipos existen?" (Turn 2). The history contains all three image markers, causing the LLM to reference "las tres imágenes mostradas" when they are not visible.

- **Example 3**: User uploads image asking for identification (Turn 1). System responds with `[IMAGEN DE CONSULTA DEL USUARIO]` and `[IMAGEN RECUPERADA 1]` in history. User asks "¿Dónde se encuentra este tejido?" (Turn 2 - text only). The LLM incorrectly references "la imagen que subiste" because the marker is still in the history.

- **Edge case**: User makes text query "¿Qué es histología?" (Turn 1 - no images). User makes another text query "Explica más" (Turn 2). The history contains no image markers, so the expected behavior is to include the full history unchanged.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Image queries (`tipo_consulta == 'imagen'`) must continue to include the full unfiltered conversation history with all image reference markers
- When `contexto_memoria` is empty or None, the system must continue to construct messages without any conversation history
- When `contexto_memoria` contains only text (no image reference markers), the system must continue to include the full conversation history as-is
- The structure and format of the `user_content` list returned by `_construir_mensaje_usuario` must remain unchanged
- The text content of conversation history (excluding image markers) must be preserved in text queries

**Scope:**
All inputs where `tipo_consulta == 'imagen'` OR `contexto_memoria` is empty/None OR `contexto_memoria` contains no image markers should be completely unaffected by this fix. This includes:
- Image query message construction with full history
- Empty history handling
- Text-only history (no image markers) handling
- The fallback behavior when all images fail to load in image queries

## Hypothesized Root Cause

Based on the bug description and code analysis, the root cause is:

1. **No Filtering Logic**: The `_construir_mensaje_usuario` method constructs the `historial` string directly from `state.get("contexto_memoria")` without any conditional filtering based on `tipo_consulta`. Lines 1469-1471 show:
   ```python
   if state.get("contexto_memoria"):
       historial = f"\n========================================\nHISTORIAL DE CONVERSACIÓN RELEVANTE:\n{state['contexto_memoria']}\n========================================\n"
   ```

2. **Image Markers in Memory**: The conversation memory system stores the complete conversation history including image reference markers like `[IMAGEN RECUPERADA 1: filename.jpg]` and `[IMAGEN DE CONSULTA DEL USUARIO]`. These markers are useful for image queries but misleading for text queries.

3. **Type-Agnostic History Construction**: The history construction happens before the `if tipo_consulta == 'texto':` branch (line 1476), meaning the same unfiltered history is used for both text and image queries.

## Correctness Properties

Property 1: Bug Condition - Filter Image References in Text Queries

_For any_ input where `tipo_consulta == 'texto'` AND `contexto_memoria` contains image reference markers (`[IMAGEN RECUPERADA` or `[IMAGEN DE CONSULTA DEL USUARIO]`), the fixed `_construir_mensaje_usuario` function SHALL remove all image reference markers from the conversation history before including it in the user message, preserving only the textual content of the conversation.

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation - Image Query History Unchanged

_For any_ input where `tipo_consulta == 'imagen'` OR `contexto_memoria` is empty/None OR `contexto_memoria` contains no image reference markers, the fixed `_construir_mensaje_usuario` function SHALL produce exactly the same result as the original function, preserving all existing behavior for image queries and non-image-containing histories.

**Validates: Requirements 3.1, 3.2, 3.3**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `muvera_test.py`

**Function**: `_construir_mensaje_usuario` (lines 1456-1600)

**Specific Changes**:

1. **Add Image Reference Filtering Helper**: Create a helper method to filter image reference markers from conversation history:
   - Method name: `_filtrar_referencias_imagenes`
   - Input: conversation history string
   - Output: filtered string with image markers removed
   - Pattern matching: Remove lines containing `[IMAGEN RECUPERADA` and `[IMAGEN DE CONSULTA DEL USUARIO]`

2. **Conditional History Filtering**: Modify the history construction logic (lines 1469-1471) to apply filtering when `tipo_consulta == 'texto'`:
   - Check if `tipo_consulta == 'texto'`
   - If true, apply `_filtrar_referencias_imagenes` to `contexto_memoria` before constructing `historial`
   - If false (image query), use `contexto_memoria` as-is

3. **Preserve Textual Content**: Ensure the filtering only removes image marker lines, not the surrounding conversational text:
   - Use regex or string operations to identify and remove only marker lines
   - Preserve line breaks and formatting of remaining text

4. **Handle Edge Cases**: Ensure the fix handles:
   - Empty `contexto_memoria` (already handled by existing `if state.get("contexto_memoria")` check)
   - History with no image markers (filtering should be a no-op)
   - History with multiple image markers (all should be removed)
   - History with mixed text and image markers (only markers removed)

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that construct AgentState objects with `contexto_memoria` containing image reference markers, call `_construir_mensaje_usuario` with `tipo_consulta='texto'` on the UNFIXED code, and assert that image markers are present in the output (demonstrating the bug).

**Test Cases**:
1. **Single Image Marker Test**: Create state with `contexto_memoria` containing `[IMAGEN RECUPERADA 1: test.jpg]`, call with `tipo_consulta='texto'` (will fail on unfixed code - markers present)
2. **Multiple Image Markers Test**: Create state with history containing 3 `[IMAGEN RECUPERADA]` markers, call with `tipo_consulta='texto'` (will fail on unfixed code - all markers present)
3. **User Image Marker Test**: Create state with `[IMAGEN DE CONSULTA DEL USUARIO]` in history, call with `tipo_consulta='texto'` (will fail on unfixed code - marker present)
4. **Mixed Content Test**: Create state with text and image markers interleaved, call with `tipo_consulta='texto'` (will fail on unfixed code - markers present, text preserved)

**Expected Counterexamples**:
- Image reference markers appear in the constructed message for text queries
- The LLM receives misleading context suggesting images are present when they are not
- Possible causes: no filtering logic, type-agnostic history construction

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := _construir_mensaje_usuario_fixed(input.state, input.tipo_consulta)
  ASSERT NOT contains_image_markers(result)
  ASSERT contains_textual_content(result, input.state["contexto_memoria"])
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT _construir_mensaje_usuario_original(input.state, input.tipo_consulta) 
         = _construir_mensaje_usuario_fixed(input.state, input.tipo_consulta)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for image queries and empty/text-only histories, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Image Query Preservation**: Observe that image queries with history containing markers work correctly on unfixed code, then write test to verify this continues after fix (markers should remain for image queries)
2. **Empty History Preservation**: Observe that empty `contexto_memoria` works correctly on unfixed code, then write test to verify this continues after fix
3. **Text-Only History Preservation**: Observe that history with no image markers works correctly on unfixed code, then write test to verify this continues after fix
4. **Message Structure Preservation**: Verify that the structure of the returned `user_content` list (dict with 'type' and 'text' keys) remains unchanged

### Unit Tests

- Test filtering helper method with various image marker patterns
- Test text query with single image marker in history (marker removed)
- Test text query with multiple image markers in history (all markers removed)
- Test text query with no image markers in history (no change)
- Test image query with image markers in history (markers preserved)
- Test empty history handling for both query types
- Test that textual content is preserved when markers are removed

### Property-Based Tests

- Generate random conversation histories with varying numbers of image markers and verify they are filtered for text queries
- Generate random conversation histories without image markers and verify they are unchanged for text queries
- Generate random conversation histories and verify image queries always preserve the full history
- Test that filtering is idempotent (applying twice produces same result as applying once)

### Integration Tests

- Test full conversation flow: image query (Turn 1) → text query (Turn 2) → verify Turn 2 has no image markers
- Test multi-turn conversation with alternating image and text queries
- Test that LLM responses for text queries after image queries no longer reference images inappropriately
