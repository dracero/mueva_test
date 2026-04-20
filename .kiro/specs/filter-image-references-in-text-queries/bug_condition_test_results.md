# Bug Condition Exploration Test Results

## Test Execution Summary

**Date**: Task 1 Execution
**Status**: ✅ Tests FAILED as expected (confirming bug exists)
**Test File**: `test_bug_condition_image_references.py`

## Purpose

These tests were written to confirm the bug exists on UNFIXED code. The tests encode the **expected behavior** (image markers should be filtered out in text queries) and are designed to FAIL on the current implementation, proving that the bug is present.

## Test Results

### Property-Based Test

**Test**: `test_property_bug_condition_image_markers_filtered_in_text_queries`

**Result**: ❌ FAILED (as expected)

**Counterexample Found by Hypothesis**:
```python
contexto_memoria='Usuario: 0000000000\n[IMAGEN RECUPERADA 1: aaaaa.jpg]\nAsistente: 0000000000'
```

**Finding**: When `tipo_consulta='texto'` and `contexto_memoria` contains `[IMAGEN RECUPERADA 1: aaaaa.jpg]`, the marker is present in the output text. This confirms the bug - image markers are NOT being filtered.

### Concrete Test Cases

#### 1. Single Retrieved Image Marker
**Test**: `test_bug_condition_single_retrieved_image_marker`

**Result**: ❌ FAILED (as expected)

**Input**:
```python
contexto_memoria = 'Usuario: ¿Qué estructura es esta?\n[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]\nAsistente: Es un epitelio estratificado.'
tipo_consulta = 'texto'
```

**Finding**: The marker `[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]` appears in the conversation history section of the text query output.

**Validates**: Requirement 2.1

---

#### 2. Multiple Retrieved Image Markers
**Test**: `test_bug_condition_multiple_retrieved_image_markers`

**Result**: ❌ FAILED (as expected)

**Input**:
```python
contexto_memoria = '''Usuario: Muéstrame ejemplos de epitelio
[IMAGEN RECUPERADA 1: arch2_p1_img1.jpg]
[IMAGEN RECUPERADA 2: arch2_p2_img2.jpg]
[IMAGEN RECUPERADA 3: arch2_p3_img3.jpg]
Asistente: Aquí hay tres ejemplos de epitelio.'''
tipo_consulta = 'texto'
```

**Finding**: All three image markers (`[IMAGEN RECUPERADA 1]`, `[IMAGEN RECUPERADA 2]`, `[IMAGEN RECUPERADA 3]`) appear in the conversation history section of the text query output.

**Validates**: Requirement 2.1

---

#### 3. User Query Image Marker
**Test**: `test_bug_condition_user_query_image_marker`

**Result**: ❌ FAILED (as expected)

**Input**:
```python
contexto_memoria = '''Usuario: ¿Qué es esto?
[IMAGEN DE CONSULTA DEL USUARIO]
[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]
Asistente: Es un epitelio estratificado.'''
tipo_consulta = 'texto'
```

**Finding**: Both markers (`[IMAGEN DE CONSULTA DEL USUARIO]` and `[IMAGEN RECUPERADA 1]`) appear in the conversation history section of the text query output.

**Validates**: Requirements 2.1, 2.2

---

#### 4. Mixed Markers and Text
**Test**: `test_bug_condition_mixed_markers_and_text`

**Result**: ❌ FAILED (as expected)

**Input**:
```python
contexto_memoria = '''Usuario: ¿Qué estructura es esta?
[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]
Asistente: Es un epitelio estratificado plano.
Usuario: ¿Puedes mostrar más ejemplos?
[IMAGEN RECUPERADA 2: arch2_p6_img6.jpg]
[IMAGEN RECUPERADA 3: arch2_p7_img7.jpg]
Asistente: Aquí hay más ejemplos de epitelio.'''
tipo_consulta = 'texto'
```

**Finding**: All image markers are present in the output. However, the textual content ('epitelio estratificado plano', 'más ejemplos de epitelio') is also preserved, which is correct.

**Validates**: Requirements 2.1, 2.2

---

## Root Cause Confirmation

The test failures confirm the hypothesized root cause from the design document:

1. **No Filtering Logic**: The `_construir_mensaje_usuario` method includes `contexto_memoria` directly in the `historial` string without any conditional filtering based on `tipo_consulta`.

2. **Image Markers in Memory**: The conversation memory system stores complete history including image reference markers like `[IMAGEN RECUPERADA N: filename.jpg]` and `[IMAGEN DE CONSULTA DEL USUARIO]`.

3. **Type-Agnostic History Construction**: The history construction happens before the query type branching, so the same unfiltered history is used for both text and image queries.

## Expected Behavior After Fix

When the fix is implemented, these same tests should PASS, confirming that:
- Image reference markers are filtered out when `tipo_consulta='texto'`
- Textual content of the conversation is preserved
- The bug is resolved

## Next Steps

1. ✅ Task 1 Complete: Bug condition exploration test written and run on unfixed code
2. ⏭️ Task 2: Write preservation property tests (before implementing fix)
3. ⏭️ Task 3: Implement the fix
4. ⏭️ Task 3.3: Re-run these tests - they should PASS after the fix
