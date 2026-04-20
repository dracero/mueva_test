"""
Bug Condition Exploration Test for Image References in Text Queries

**Property 1: Bug Condition** - Image References Present in Text Query History

This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

The test encodes the expected behavior - it will validate the fix when it passes after implementation.

**Validates: Requirements 2.1, 2.2**
"""
import os
import sys
import pytest
from hypothesis import given, strategies as st, settings, Phase, HealthCheck

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from muvera_test import SistemaRAGColPaliPuro


@pytest.fixture
def sistema():
    """Fixture que crea una instancia del sistema para testing"""
    return SistemaRAGColPaliPuro()


# Strategy for generating conversation history with image markers
@st.composite
def history_with_image_markers(draw):
    """Generate conversation history containing image reference markers"""
    # Choose which type of markers to include
    marker_types = draw(st.lists(
        st.sampled_from([
            'retrieved',  # [IMAGEN RECUPERADA N: filename]
            'user_query'  # [IMAGEN DE CONSULTA DEL USUARIO]
        ]),
        min_size=1,
        max_size=5
    ))
    
    history_parts = []
    
    for marker_type in marker_types:
        # Add some conversational text before the marker
        text_before = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'), min_codepoint=32, max_codepoint=126),
            min_size=10,
            max_size=100
        ))
        history_parts.append(f"Usuario: {text_before}")
        
        # Add the image marker
        if marker_type == 'retrieved':
            img_num = draw(st.integers(min_value=1, max_value=10))
            filename = draw(st.text(
                alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=97, max_codepoint=122),
                min_size=5,
                max_size=15
            ))
            marker = f"[IMAGEN RECUPERADA {img_num}: {filename}.jpg]"
        else:
            marker = "[IMAGEN DE CONSULTA DEL USUARIO]"
        
        history_parts.append(marker)
    
    # Add some final conversational text
    text_after = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'), min_codepoint=32, max_codepoint=126),
        min_size=10,
        max_size=100
    ))
    history_parts.append(f"Asistente: {text_after}")
    
    return "\n".join(history_parts)


@given(contexto_memoria=history_with_image_markers())
@settings(
    max_examples=50,
    phases=[Phase.generate, Phase.target],
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_bug_condition_image_markers_filtered_in_text_queries(sistema, contexto_memoria):
    """
    Property 1: Bug Condition - Image References Filtered in Text Query History
    
    For any input where tipo_consulta='texto' AND contexto_memoria contains image reference markers,
    the _construir_mensaje_usuario function SHOULD remove all image reference markers from the
    conversation history before including it in the user message.
    
    **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    **EXPECTED OUTCOME**: Test FAILS (markers are present in output, proving the bug)
    
    **Validates: Requirements 2.1, 2.2**
    """
    # Arrange: Create state with text query and history containing image markers
    state = {
        'consulta_usuario': '¿Cuáles son las características?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Contexto sobre histopatología.',
        'contexto_memoria': contexto_memoria
    }
    
    # Act: Call _construir_mensaje_usuario with tipo_consulta='texto'
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    
    # Extract the text content from the message
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Assert: Image markers should NOT be present in the output (expected behavior)
    # This assertion will FAIL on unfixed code, proving the bug exists
    assert '[IMAGEN RECUPERADA' not in texto_completo, \
        f"Image marker '[IMAGEN RECUPERADA' found in text query output. This confirms the bug exists."
    
    assert '[IMAGEN DE CONSULTA DEL USUARIO]' not in texto_completo, \
        f"Image marker '[IMAGEN DE CONSULTA DEL USUARIO]' found in text query output. This confirms the bug exists."


# Concrete test cases for specific scenarios
def test_bug_condition_single_retrieved_image_marker(sistema):
    """
    Concrete test: Single retrieved image marker in history
    
    **EXPECTED OUTCOME**: Test FAILS (marker present in output)
    **Validates: Requirements 2.1**
    """
    state = {
        'consulta_usuario': '¿Qué características tiene?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio es un tejido.',
        'contexto_memoria': 'Usuario: ¿Qué estructura es esta?\n[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]\nAsistente: Es un epitelio estratificado.'
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    assert '[IMAGEN RECUPERADA' not in texto_completo, \
        "Image marker should be filtered out in text queries"


def test_bug_condition_multiple_retrieved_image_markers(sistema):
    """
    Concrete test: Multiple retrieved image markers in history
    
    **EXPECTED OUTCOME**: Test FAILS (markers present in output)
    **Validates: Requirements 2.1**
    """
    state = {
        'consulta_usuario': '¿Qué tipos existen?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Existen varios tipos de epitelio.',
        'contexto_memoria': '''Usuario: Muéstrame ejemplos de epitelio
[IMAGEN RECUPERADA 1: arch2_p1_img1.jpg]
[IMAGEN RECUPERADA 2: arch2_p2_img2.jpg]
[IMAGEN RECUPERADA 3: arch2_p3_img3.jpg]
Asistente: Aquí hay tres ejemplos de epitelio.'''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    assert '[IMAGEN RECUPERADA' not in texto_completo, \
        "All image markers should be filtered out in text queries"


def test_bug_condition_user_query_image_marker(sistema):
    """
    Concrete test: User query image marker in history
    
    **EXPECTED OUTCOME**: Test FAILS (marker present in output)
    **Validates: Requirements 2.2**
    """
    state = {
        'consulta_usuario': '¿Dónde se encuentra este tejido?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio se encuentra en varias ubicaciones.',
        'contexto_memoria': '''Usuario: ¿Qué es esto?
[IMAGEN DE CONSULTA DEL USUARIO]
[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]
Asistente: Es un epitelio estratificado.'''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    assert '[IMAGEN DE CONSULTA DEL USUARIO]' not in texto_completo, \
        "User query image marker should be filtered out in text queries"


def test_bug_condition_mixed_markers_and_text(sistema):
    """
    Concrete test: Mixed image markers and conversational text
    
    Verifies that textual content is preserved while markers are removed.
    
    **EXPECTED OUTCOME**: Test FAILS (markers present in output)
    **Validates: Requirements 2.1, 2.2**
    """
    state = {
        'consulta_usuario': '¿Cuál es la función?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio tiene funciones de protección.',
        'contexto_memoria': '''Usuario: ¿Qué estructura es esta?
[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]
Asistente: Es un epitelio estratificado plano.
Usuario: ¿Puedes mostrar más ejemplos?
[IMAGEN RECUPERADA 2: arch2_p6_img6.jpg]
[IMAGEN RECUPERADA 3: arch2_p7_img7.jpg]
Asistente: Aquí hay más ejemplos de epitelio.'''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Image markers should be removed
    assert '[IMAGEN RECUPERADA' not in texto_completo, \
        "Image markers should be filtered out in text queries"
    
    # But textual content should be preserved
    assert 'epitelio estratificado plano' in texto_completo, \
        "Textual content should be preserved when filtering markers"
    assert 'más ejemplos de epitelio' in texto_completo, \
        "Textual content should be preserved when filtering markers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
