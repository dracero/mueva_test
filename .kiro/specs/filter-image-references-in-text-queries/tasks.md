# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Image References Present in Text Query History
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to concrete failing cases where `tipo_consulta='texto'` and `contexto_memoria` contains image reference markers
  - Test that `_construir_mensaje_usuario` with `tipo_consulta='texto'` and history containing `[IMAGEN RECUPERADA` or `[IMAGEN DE CONSULTA DEL USUARIO]` markers produces output WITHOUT those markers (from Bug Condition in design)
  - The test assertions should match the Expected Behavior Properties from design: no image markers in text query output
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists, markers are present in output)
  - Document counterexamples found to understand root cause
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.2_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Image Query and Non-Image History Behavior
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs:
    - Image queries (`tipo_consulta='imagen'`) with image markers in history
    - Text queries with empty `contexto_memoria`
    - Text queries with `contexto_memoria` containing no image markers
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Fix for image references in text query history

  - [x] 3.1 Implement the filtering helper method
    - Add `_filtrar_referencias_imagenes` method to filter image reference markers from conversation history
    - Use regex or string operations to remove lines containing `[IMAGEN RECUPERADA` and `[IMAGEN DE CONSULTA DEL USUARIO]`
    - Preserve textual content and line breaks
    - Handle edge cases: empty input, no markers, multiple markers
    - _Bug_Condition: isBugCondition(input) where tipo_consulta == 'texto' AND contexto_memoria contains '[IMAGEN RECUPERADA' OR '[IMAGEN DE CONSULTA DEL USUARIO]'_
    - _Expected_Behavior: Image markers removed from history, textual content preserved_
    - _Preservation: Image queries and empty/text-only histories unchanged_
    - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3_

  - [x] 3.2 Apply conditional filtering in _construir_mensaje_usuario
    - Modify history construction logic (lines 1469-1471) to apply filtering when `tipo_consulta == 'texto'`
    - Check if `tipo_consulta == 'texto'` before constructing `historial`
    - If true, apply `_filtrar_referencias_imagenes` to `contexto_memoria`
    - If false (image query), use `contexto_memoria` as-is
    - Preserve existing structure of `user_content` list
    - _Bug_Condition: isBugCondition(input) from design_
    - _Expected_Behavior: expectedBehavior(result) from design - no image markers in text query messages_
    - _Preservation: Preservation Requirements from design - image queries unchanged_
    - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3_

  - [x] 3.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Image References Filtered in Text Query History
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed - no image markers in text query output)
    - _Requirements: Expected Behavior Properties from design - 2.1, 2.2_

  - [x] 3.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Image Query and Non-Image History Behavior
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: Preservation Requirements from design - 3.1, 3.2, 3.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
