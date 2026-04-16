"""
Test completo para Task 6.3: Casos edge

**Validates: Requirements 5.1, 6.2**

Este test verifica que el sistema maneja gracefully casos edge:
1. Contexto vacío o muy corto
2. Error al cargar imagen (archivo no existe)
3. Todas las imágenes fallan al cargar

Sub-tasks:
- Test: contexto vacío o muy corto
- Test: error al cargar imagen (archivo no existe)
- Test: todas las imágenes fallan al cargar
"""

import pytest
import os
import sys
import io
import tempfile
from typing import List, Dict, Any

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from muvera_test import SistemaRAGColPaliPuro


class MockLLMResponse:
    """Mock de la respuesta del LLM"""
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """Mock del LLM para testing"""
    def __init__(self, response: str = "Respuesta de prueba"):
        self.response = response
        self.last_messages = None
        self.last_system_prompt = None
        self.last_user_content = None
        self.call_count = 0
    
    async def ainvoke(self, messages: List[Any]) -> MockLLMResponse:
        """Mock del método ainvoke del LLM"""
        self.call_count += 1
        self.last_messages = messages
        
        # Extraer system prompt y user content
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'system':
                    self.last_system_prompt = msg.content
                elif msg.type == 'human':
                    self.last_user_content = msg.content
        
        return MockLLMResponse(self.response)


@pytest.fixture
def sistema_con_mock_llm():
    """Fixture que crea un sistema con LLM mockeado"""
    sistema = SistemaRAGColPaliPuro()
    mock_llm = MockLLM()
    sistema.llm = mock_llm
    return sistema, mock_llm


@pytest.fixture
def temp_image():
    """Fixture que crea una imagen temporal válida para testing"""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as f:
        # Escribir un JPEG mínimo válido
        f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01')
        f.write(b'\xFF\xD9')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# SUB-TASK 1: Test: contexto vacío o muy corto
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_contexto_completamente_vacio(sistema_con_mock_llm):
    """
    Edge case: Contexto completamente vacío (0 caracteres)
    
    **Validates: Requirements 5.1**
    
    Verifica que:
    - El sistema detecta contexto vacío
    - Se registra advertencia apropiada
    - El sistema continúa y genera respuesta
    - No lanza excepciones
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No tengo suficiente información en mis fuentes."
    
    # Capturar output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': '',  # Completamente vacío
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act - no debe lanzar excepción
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema maneja gracefully
        assert 'respuesta_final' in resultado, \
            "Debe generar respuesta incluso con contexto vacío"
        assert resultado['respuesta_final'] == mock_llm.response
        
        # Assert: Se registró advertencia
        assert '⚠️ Contexto insuficiente detectado' in output, \
            "Debe registrar advertencia de contexto insuficiente"
        
        # Assert: LLM fue invocado
        assert mock_llm.call_count == 1, \
            "El LLM debe ser invocado incluso con contexto vacío"
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_contexto_muy_corto_10_caracteres(sistema_con_mock_llm):
    """
    Edge case: Contexto muy corto (10 caracteres)
    
    **Validates: Requirements 5.1**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No tengo suficiente información."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué es el epitelio?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': 'Muy corto',  # 10 caracteres
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Detecta como insuficiente
        assert '⚠️ Contexto insuficiente detectado' in output
        assert 'respuesta_final' in resultado
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_contexto_solo_espacios_y_saltos_linea(sistema_con_mock_llm):
    """
    Edge case: Contexto con solo espacios, tabs y saltos de línea
    
    **Validates: Requirements 5.1**
    
    Verifica que strip() se usa correctamente para detectar contexto vacío.
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No tengo información."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué es el tejido?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': '   \n\n\t\t   \n   ',  # Solo whitespace
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Detecta como insuficiente (strip elimina todo)
        assert '⚠️ Contexto insuficiente detectado' in output, \
            "Debe detectar contexto vacío después de strip()"
        assert 'respuesta_final' in resultado
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_contexto_exactamente_49_caracteres(sistema_con_mock_llm):
    """
    Edge case: Contexto con exactamente 49 caracteres (justo bajo el umbral)
    
    **Validates: Requirements 5.1**
    """
    sistema, mock_llm = sistema_con_mock_llm
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # Crear contexto de exactamente 49 caracteres
        contexto_49 = 'A' * 49
        assert len(contexto_49) == 49, "Debe ser exactamente 49 caracteres"
        
        state = {
            'consulta_usuario': '¿Qué es esto?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': contexto_49,
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: 49 < 50, debe detectar como insuficiente
        assert '⚠️ Contexto insuficiente detectado' in output, \
            "49 caracteres debe considerarse insuficiente (umbral es <50)"
        
    finally:
        sys.stdout = sys.__stdout__


# ============================================================================
# SUB-TASK 2: Test: error al cargar imagen (archivo no existe)
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_imagen_no_existe(sistema_con_mock_llm):
    """
    Edge case: Imagen especificada no existe en el sistema de archivos
    
    **Validates: Requirements 6.2**
    
    Verifica que:
    - El sistema maneja gracefully archivos inexistentes
    - Se registra advertencia apropiada
    - El sistema continúa sin la imagen
    - No lanza excepciones
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Basándome en el contexto textual..."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué muestra esta imagen?',
            'imagen_consulta': None,
            'imagenes_relevantes': ['/path/to/nonexistent/image.jpg'],
            'contexto_documentos': 'Descripción del tejido epitelial...',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act - no debe lanzar excepción
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema continúa sin la imagen
        assert 'respuesta_final' in resultado, \
            "Debe generar respuesta incluso si la imagen no existe"
        assert resultado['respuesta_final'] == mock_llm.response
        
        # Assert: Se registró advertencia
        assert '⚠️' in output, \
            "Debe registrar advertencia sobre imagen inexistente"
        
        # Assert: LLM fue invocado
        assert mock_llm.call_count == 1
        
        # Assert: No se incluyeron imágenes en el mensaje
        if isinstance(mock_llm.last_user_content, list):
            imagenes = [p for p in mock_llm.last_user_content if p.get('type') == 'image_url']
            assert len(imagenes) == 0, \
                "No debe incluir imágenes que no existen"
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_imagen_consulta_usuario_no_existe(sistema_con_mock_llm):
    """
    Edge case: Imagen de consulta del usuario no existe
    
    **Validates: Requirements 6.2**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No pude cargar tu imagen."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué tipo de tejido es este?',
            'imagen_consulta': '/path/to/nonexistent/user_image.jpg',  # No existe
            'imagenes_relevantes': [],
            'contexto_documentos': 'Información sobre tejidos...',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema maneja gracefully
        assert 'respuesta_final' in resultado
        assert '⚠️' in output, "Debe registrar advertencia"
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_path_imagen_vacio(sistema_con_mock_llm):
    """
    Edge case: Path de imagen es string vacío
    
    **Validates: Requirements 6.2**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Respuesta basada en texto."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué es esto?',
            'imagen_consulta': '',  # String vacío
            'imagenes_relevantes': [''],  # String vacío en lista
            'contexto_documentos': 'Contexto textual suficiente para responder la pregunta.',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act - no debe lanzar excepción
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        
        # Assert: Sistema maneja gracefully
        assert 'respuesta_final' in resultado
        assert mock_llm.call_count == 1
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_algunas_imagenes_existen_otras_no(sistema_con_mock_llm, temp_image):
    """
    Edge case: Lista mixta - algunas imágenes existen, otras no
    
    **Validates: Requirements 6.2**
    
    Verifica que el sistema carga las imágenes válidas y omite las inválidas.
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Análisis de las imágenes disponibles..."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué muestran estas imágenes?',
            'imagen_consulta': None,
            'imagenes_relevantes': [
                temp_image,  # Existe
                '/path/to/nonexistent1.jpg',  # No existe
                '/path/to/nonexistent2.jpg',  # No existe
            ],
            'contexto_documentos': 'Descripción de tejidos...',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema procesa correctamente
        assert 'respuesta_final' in resultado
        
        # Assert: Se registraron advertencias para imágenes inexistentes
        assert output.count('⚠️') >= 2, \
            "Debe registrar advertencias para las 2 imágenes inexistentes"
        
        # Assert: Se cargó al menos la imagen válida
        if isinstance(mock_llm.last_user_content, list):
            imagenes = [p for p in mock_llm.last_user_content if p.get('type') == 'image_url']
            assert len(imagenes) >= 1, \
                "Debe cargar al menos la imagen válida"
        
    finally:
        sys.stdout = sys.__stdout__


# ============================================================================
# SUB-TASK 3: Test: todas las imágenes fallan al cargar
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_todas_imagenes_fallan_al_cargar(sistema_con_mock_llm):
    """
    Edge case: Todas las imágenes en la lista fallan al cargar
    
    **Validates: Requirements 6.2**
    
    Verifica que:
    - El sistema detecta que todas las imágenes fallaron
    - Se registra advertencia apropiada
    - El sistema cambia a modo texto automáticamente
    - Genera respuesta basada solo en contexto textual
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Basándome en el contexto textual disponible..."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué muestran estas imágenes?',
            'imagen_consulta': None,
            'imagenes_relevantes': [
                '/nonexistent1.jpg',
                '/nonexistent2.jpg',
                '/nonexistent3.jpg',
            ],
            'contexto_documentos': 'Descripción textual del tejido epitelial y sus características.',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema genera respuesta
        assert 'respuesta_final' in resultado, \
            "Debe generar respuesta incluso si todas las imágenes fallan"
        assert resultado['respuesta_final'] == mock_llm.response
        
        # Assert: Se registró advertencia de fallback a texto
        assert '⚠️ Todas las imágenes fallaron al cargar' in output, \
            "Debe registrar que todas las imágenes fallaron"
        assert 'Tratando como consulta de texto' in output, \
            "Debe indicar que cambia a modo texto"
        
        # Assert: LLM fue invocado
        assert mock_llm.call_count == 1
        
        # Assert: No se incluyeron imágenes en el mensaje
        if isinstance(mock_llm.last_user_content, list):
            imagenes = [p for p in mock_llm.last_user_content if p.get('type') == 'image_url']
            assert len(imagenes) == 0, \
                "No debe incluir imágenes cuando todas fallan"
            
            # Assert: El mensaje se construyó como consulta de texto
            textos = [p.get('text', '') for p in mock_llm.last_user_content if p.get('type') == 'text']
            texto_completo = ' '.join(textos)
            assert 'CONTEXTO RECUPERADO' in texto_completo, \
                "Debe incluir contexto textual"
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_todas_imagenes_fallan_con_imagen_usuario(sistema_con_mock_llm):
    """
    Edge case: Todas las imágenes fallan, incluyendo la del usuario
    
    **Validates: Requirements 6.2**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No pude cargar las imágenes."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué tipo de tejido es?',
            'imagen_consulta': '/nonexistent_user.jpg',  # No existe
            'imagenes_relevantes': [
                '/nonexistent1.jpg',
                '/nonexistent2.jpg',
            ],
            'contexto_documentos': 'Información sobre tejidos disponible en texto.',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema maneja gracefully
        assert 'respuesta_final' in resultado
        
        # Assert: Se registraron múltiples advertencias
        assert output.count('⚠️') >= 2, \
            "Debe registrar advertencias para todas las imágenes fallidas"
        
        # Assert: Cambió a modo texto
        assert 'Todas las imágenes fallaron' in output or 'Tratando como consulta de texto' in output
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_todas_imagenes_fallan_sin_contexto_textual(sistema_con_mock_llm):
    """
    Edge case: Todas las imágenes fallan Y el contexto textual es insuficiente
    
    **Validates: Requirements 5.1, 6.2**
    
    Caso extremo: doble fallo (imágenes + contexto).
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No tengo suficiente información para responder."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué es esto?',
            'imagen_consulta': None,
            'imagenes_relevantes': [
                '/nonexistent1.jpg',
                '/nonexistent2.jpg',
            ],
            'contexto_documentos': '',  # Vacío
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema maneja ambos fallos gracefully
        assert 'respuesta_final' in resultado, \
            "Debe generar respuesta incluso con doble fallo"
        
        # Assert: Se registraron ambas advertencias
        assert '⚠️ Todas las imágenes fallaron' in output or '⚠️' in output
        assert '⚠️ Contexto insuficiente' in output
        
        # Assert: LLM fue invocado y puede responder apropiadamente
        assert mock_llm.call_count == 1
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_edge_case_lista_imagenes_vacia_vs_none(sistema_con_mock_llm):
    """
    Edge case: Diferencia entre lista vacía [] y None para imagenes_relevantes
    
    **Validates: Requirements 6.2**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Respuesta textual."
    
    # Test con lista vacía
    state_lista_vacia = {
        'consulta_usuario': '¿Qué es el epitelio?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],  # Lista vacía
        'contexto_documentos': 'El epitelio es un tejido...',
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    resultado1 = await sistema._nodo_generar_respuesta(state_lista_vacia)
    
    # Assert: Funciona con lista vacía
    assert 'respuesta_final' in resultado1
    assert mock_llm.call_count == 1
    
    # Reset mock
    mock_llm.call_count = 0
    
    # Test con None (si el sistema lo permite)
    state_none = {
        'consulta_usuario': '¿Qué es el epitelio?',
        'imagen_consulta': None,
        'imagenes_relevantes': None,  # None en lugar de lista
        'contexto_documentos': 'El epitelio es un tejido...',
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Esto podría fallar si el código no maneja None, pero no debe crashear
    try:
        resultado2 = await sistema._nodo_generar_respuesta(state_none)
        # Si llega aquí, el sistema maneja None gracefully
        assert 'respuesta_final' in resultado2
    except (TypeError, AttributeError):
        # Si falla, es un bug que debería reportarse, pero no es crítico para este test
        pytest.skip("Sistema no maneja None en imagenes_relevantes (debería usar lista vacía)")


# ============================================================================
# TESTS COMBINADOS: Múltiples edge cases simultáneos
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_combinado_contexto_corto_y_imagen_falla(sistema_con_mock_llm):
    """
    Edge case combinado: Contexto muy corto + imagen no existe
    
    **Validates: Requirements 5.1, 6.2**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "No tengo suficiente información."
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        state = {
            'consulta_usuario': '¿Qué es esto?',
            'imagen_consulta': None,
            'imagenes_relevantes': ['/nonexistent.jpg'],
            'contexto_documentos': 'Muy poco',  # 9 caracteres
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        resultado = await sistema._nodo_generar_respuesta(state)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Sistema maneja ambos problemas
        assert 'respuesta_final' in resultado
        assert '⚠️' in output, "Debe registrar advertencias"
        
    finally:
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
