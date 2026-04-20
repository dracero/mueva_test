"""
Preservation Property Tests for Image References in Text Queries Bugfix

**Property 2: Preservation** - Image Query and Non-Image History Behavior

These tests MUST PASS on unfixed code - they establish the baseline behavior to preserve.

**IMPORTANT**: Follow observation-first methodology
- Observe behavior on UNFIXED code for non-buggy inputs
- Write property-based tests capturing observed behavior patterns
- Run tests on UNFIXED code
- **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)

**Validates: Requirements 3.1, 3.2, 3.3**
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


# Strategy for generating conversation history WITHOUT image markers
@st.composite
def history_without_image_markers(draw):
    """Generate conversation history containing only text (no image markers)"""
    num_exchanges = draw(st.integers(min_value=1, max_value=5))
    history_parts = []
    
    for _ in range(num_exchanges):
        user_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'), min_codepoint=32, max_codepoint=126),
            min_size=10,
            max_size=100
        ))
        assistant_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'), min_codepoint=32, max_codepoint=126),
            min_size=10,
            max_size=100
        ))
        history_parts.append(f"Usuario: {user_text}")
        history_parts.append(f"Asistente: {assistant_text}")
    
    return "\n".join(history_parts)


@given(contexto_memoria=history_with_image_markers())
@settings(
    max_examples=50,
    phases=[Phase.generate, Phase.target],
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_preservation_image_queries_keep_markers(sistema, contexto_memoria):
    """
    Property 2.1: Preservation - Image Queries Keep Image Markers
    
    For any input where tipo_consulta='imagen' AND contexto_memoria contains image reference markers,
    the _construir_mensaje_usuario function MUST preserve all image reference markers in the
    conversation history (unchanged behavior).
    
    **EXPECTED OUTCOME**: Test PASSES (markers are preserved for image queries)
    
    **Validates: Requirement 3.1**
    """
    # Arrange: Create state with image query and history containing image markers
    state = {
        'consulta_usuario': '¿Qué estructura es esta?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Contexto sobre histopatología.',
        'contexto_memoria': contexto_memoria
    }
    
    # Act: Call _construir_mensaje_usuario with tipo_consulta='imagen'
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    
    # Extract the text content from the message
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Assert: Image markers SHOULD be present in the output for image queries
    # This is the baseline behavior that must be preserved
    if '[IMAGEN RECUPERADA' in contexto_memoria:
        assert '[IMAGEN RECUPERADA' in texto_completo, \
            "Image markers should be preserved in image query output (baseline behavior)"
    
    if '[IMAGEN DE CONSULTA DEL USUARIO]' in contexto_memoria:
        assert '[IMAGEN DE CONSULTA DEL USUARIO]' in texto_completo, \
            "User query image markers should be preserved in image query output (baseline behavior)"


@given(contexto_memoria=history_without_image_markers())
@settings(
    max_examples=50,
    phases=[Phase.generate, Phase.target],
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_preservation_text_only_history_unchanged(sistema, contexto_memoria):
    """
    Property 2.2: Preservation - Text-Only History Unchanged
    
    For any input where tipo_consulta='texto' AND contexto_memoria contains NO image reference markers,
    the _construir_mensaje_usuario function MUST include the full conversation history as-is
    (unchanged behavior).
    
    **EXPECTED OUTCOME**: Test PASSES (text-only history is preserved)
    
    **Validates: Requirement 3.3**
    """
    # Arrange: Create state with text query and history containing NO image markers
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
    
    # Assert: The full conversation history should be present in the output
    # This is the baseline behavior that must be preserved
    assert contexto_memoria in texto_completo, \
        "Text-only conversation history should be fully preserved in text query output (baseline behavior)"


def test_preservation_empty_history_text_query(sistema):
    """
    Concrete test: Empty history with text query
    
    **EXPECTED OUTCOME**: Test PASSES (empty history handled correctly)
    **Validates: Requirement 3.2**
    """
    state = {
        'consulta_usuario': '¿Qué es histología?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'La histología es el estudio de los tejidos.',
        'contexto_memoria': None
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Should not contain history section when contexto_memoria is None
    assert 'HISTORIAL DE CONVERSACIÓN RELEVANTE' not in texto_completo, \
        "Empty history should not include history section (baseline behavior)"
    
    # Should contain the query and context
    assert '¿Qué es histología?' in texto_completo
    assert 'La histología es el estudio de los tejidos' in texto_completo


def test_preservation_empty_history_image_query(sistema):
    """
    Concrete test: Empty history with image query
    
    **EXPECTED OUTCOME**: Test PASSES (empty history handled correctly)
    **Validates: Requirement 3.2**
    """
    state = {
        'consulta_usuario': '¿Qué estructura es esta?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Contexto sobre estructuras.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Should not contain history section when contexto_memoria is empty
    assert 'HISTORIAL DE CONVERSACIÓN RELEVANTE' not in texto_completo, \
        "Empty history should not include history section (baseline behavior)"
    
    # Should contain the query and context
    assert '¿Qué estructura es esta?' in texto_completo
    assert 'Contexto sobre estructuras' in texto_completo


def test_preservation_image_query_with_markers(sistema):
    """
    Concrete test: Image query with image markers in history
    
    **EXPECTED OUTCOME**: Test PASSES (markers preserved for image queries)
    **Validates: Requirement 3.1**
    """
    state = {
        'consulta_usuario': '¿Qué más puedes decirme?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Información adicional sobre epitelio.',
        'contexto_memoria': '''Usuario: ¿Qué estructura es esta?
[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]
Asistente: Es un epitelio estratificado.'''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Image markers should be preserved for image queries
    assert '[IMAGEN RECUPERADA 1: arch2_p5_img5.jpg]' in texto_completo, \
        "Image markers should be preserved in image query output (baseline behavior)"
    assert 'Es un epitelio estratificado' in texto_completo, \
        "Textual content should be preserved in image query output (baseline behavior)"


def test_preservation_text_query_without_markers(sistema):
    """
    Concrete test: Text query with text-only history (no markers)
    
    **EXPECTED OUTCOME**: Test PASSES (text-only history preserved)
    **Validates: Requirement 3.3**
    """
    state = {
        'consulta_usuario': 'Explica más sobre eso',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio tiene varias funciones.',
        'contexto_memoria': '''Usuario: ¿Qué es histología?
Asistente: La histología es el estudio de los tejidos.
Usuario: ¿Qué tipos de tejidos existen?
Asistente: Existen cuatro tipos principales: epitelial, conectivo, muscular y nervioso.'''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    
    # Full text-only history should be preserved
    assert 'La histología es el estudio de los tejidos' in texto_completo, \
        "Text-only history should be fully preserved (baseline behavior)"
    assert 'Existen cuatro tipos principales' in texto_completo, \
        "Text-only history should be fully preserved (baseline behavior)"


def test_preservation_message_structure(sistema):
    """
    Concrete test: Message structure preservation
    
    Verifies that the structure of the returned user_content list remains unchanged.
    
    **EXPECTED OUTCOME**: Test PASSES (message structure preserved)
    **Validates: Preservation of message structure**
    """
    state = {
        'consulta_usuario': '¿Qué es esto?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Contexto de prueba.',
        'contexto_memoria': 'Usuario: Pregunta anterior\nAsistente: Respuesta anterior'
    }
    
    # Test text query
    mensaje_texto = sistema._construir_mensaje_usuario(state, 'texto')
    assert isinstance(mensaje_texto, list), "Message should be a list"
    assert all(isinstance(p, dict) for p in mensaje_texto), "All parts should be dicts"
    assert all('type' in p and 'text' in p for p in mensaje_texto if p['type'] == 'text'), \
        "Text parts should have 'type' and 'text' keys"
    
    # Test image query
    mensaje_imagen = sistema._construir_mensaje_usuario(state, 'imagen')
    assert isinstance(mensaje_imagen, list), "Message should be a list"
    assert all(isinstance(p, dict) for p in mensaje_imagen), "All parts should be dicts"
    assert all('type' in p for p in mensaje_imagen), "All parts should have 'type' key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
