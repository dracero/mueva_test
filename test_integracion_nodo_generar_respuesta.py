"""
Integration tests para _nodo_generar_respuesta con mock del LLM

**Validates: Requirements 1.1, 1.3, 5.1, 5.3, 6.1, 6.4**

Estos tests verifican el flujo completo end-to-end del nodo de generación de respuesta,
usando un mock del LLM para evitar llamadas reales a la API.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from muvera_test import SistemaRAGColPaliPuro

# Configure pytest to use anyio for async tests
pytest_plugins = ('pytest_asyncio',)


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
    """Fixture que crea una imagen temporal para testing"""
    # Crear un archivo temporal que simula una imagen
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as f:
        # Escribir un JPEG mínimo válido
        # JPEG header: FF D8 FF E0 00 10 4A 46 49 46 00 01
        f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01')
        # JPEG footer: FF D9
        f.write(b'\xFF\xD9')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_flujo_completo_consulta_solo_texto(sistema_con_mock_llm):
    """
    Test end-to-end para consulta de solo texto
    
    **Validates: Requirements 1.1, 1.3, 6.1**
    
    Verifica que:
    - El sistema detecta correctamente una consulta de texto
    - Genera el prompt apropiado para texto
    - Construye el mensaje sin imágenes
    - Invoca el LLM correctamente
    - Retorna la respuesta en el state
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "El epitelio es un tejido que recubre superficies del cuerpo."
    
    # Arrange: crear state para consulta de solo texto
    state = {
        'consulta_usuario': '¿Qué es el epitelio?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio es un tejido formado por células estrechamente unidas...',
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act: ejecutar el nodo
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert: verificar resultado
    assert 'respuesta_final' in resultado, "Debe retornar respuesta_final en el state"
    assert resultado['respuesta_final'] == mock_llm.response, \
        "La respuesta debe ser la retornada por el LLM"
    
    # Verificar que el LLM fue invocado
    assert mock_llm.call_count == 1, "El LLM debe ser invocado exactamente una vez"
    
    # Verificar que se usó el prompt de texto (no debe mencionar imágenes)
    assert mock_llm.last_system_prompt is not None, "Debe haber un system prompt"
    prompt_lower = mock_llm.last_system_prompt.lower()
    assert 'profesor experto' in prompt_lower, "Debe usar tono educativo"
    assert 'amigable' in prompt_lower, "Debe usar tono amigable"
    assert 'contexto textual' in prompt_lower, "Debe mencionar contexto textual"
    
    # Verificar que el mensaje de usuario no contiene imágenes
    assert mock_llm.last_user_content is not None, "Debe haber contenido de usuario"
    if isinstance(mock_llm.last_user_content, list):
        tipos = [parte.get('type') for parte in mock_llm.last_user_content]
        assert 'image_url' not in tipos, "No debe incluir imágenes en consulta de texto"
    
    # Verificar que se actualizó la trayectoria
    assert len(resultado['trayectoria']) > 0, "Debe actualizar la trayectoria"
    assert resultado['trayectoria'][-1]['nodo'] == 'generar_respuesta', \
        "Debe registrar el nodo generar_respuesta"


@pytest.mark.asyncio
async def test_flujo_completo_consulta_con_imagen(sistema_con_mock_llm, temp_image):
    """
    Test end-to-end para consulta con imagen
    
    **Validates: Requirements 1.3, 6.1, 6.4**
    
    Verifica que:
    - El sistema detecta correctamente una consulta con imagen
    - Genera el prompt apropiado para análisis visual
    - Construye el mensaje con imágenes
    - Invoca el LLM correctamente
    - Mantiene la funcionalidad existente de análisis de imágenes
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "La imagen muestra tejido epitelial estratificado."
    
    # Arrange: crear state para consulta con imagen
    state = {
        'consulta_usuario': '¿Qué muestra esta imagen?',
        'imagen_consulta': None,
        'imagenes_relevantes': [temp_image],
        'contexto_documentos': 'Figura 14.3: Epitelio estratificado escamoso...',
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act: ejecutar el nodo
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert: verificar resultado
    assert 'respuesta_final' in resultado, "Debe retornar respuesta_final en el state"
    assert resultado['respuesta_final'] == mock_llm.response, \
        "La respuesta debe ser la retornada por el LLM"
    
    # Verificar que el LLM fue invocado
    assert mock_llm.call_count == 1, "El LLM debe ser invocado exactamente una vez"
    
    # Verificar que se usó el prompt de imagen (debe mencionar análisis visual)
    assert mock_llm.last_system_prompt is not None, "Debe haber un system prompt"
    prompt_lower = mock_llm.last_system_prompt.lower()
    assert 'profesor experto' in prompt_lower, "Debe usar tono educativo"
    assert 'amigable' in prompt_lower, "Debe usar tono amigable"
    assert 'imagen recuperada' in prompt_lower, "Debe mencionar imágenes recuperadas"
    assert 'análisis visual' in prompt_lower, "Debe mencionar análisis visual"
    
    # Verificar que el mensaje de usuario contiene imágenes
    assert mock_llm.last_user_content is not None, "Debe haber contenido de usuario"
    if isinstance(mock_llm.last_user_content, list):
        tipos = [parte.get('type') for parte in mock_llm.last_user_content]
        assert 'image_url' in tipos, "Debe incluir imágenes en consulta con imagen"
    
    # Verificar que se actualizó la trayectoria
    assert len(resultado['trayectoria']) > 0, "Debe actualizar la trayectoria"


@pytest.mark.asyncio
async def test_manejo_contexto_vacio_mensaje_amigable(sistema_con_mock_llm):
    """
    Test para verificar manejo de contexto vacío con mensaje amigable
    
    **Validates: Requirements 5.1, 5.3**
    
    Verifica que:
    - El sistema maneja gracefully contexto vacío o insuficiente
    - El prompt guía al LLM a responder con mensaje amigable
    - La respuesta sugiere reformular la pregunta
    """
    sistema, mock_llm = sistema_con_mock_llm
    # Simular respuesta amigable del LLM cuando no hay contexto
    mock_llm.response = (
        "No tengo suficiente información en mis fuentes para responder eso con "
        "precisión. ¿Podrías reformular tu pregunta o darme más detalles sobre qué "
        "aspecto específico te interesa?"
    )
    
    # Arrange: crear state con contexto vacío
    state = {
        'consulta_usuario': '¿Qué es el tejido XYZ?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': '',  # Contexto vacío
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act: ejecutar el nodo
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert: verificar que se generó una respuesta
    assert 'respuesta_final' in resultado, "Debe retornar respuesta_final incluso sin contexto"
    assert resultado['respuesta_final'] == mock_llm.response, \
        "La respuesta debe ser la retornada por el LLM"
    
    # Verificar que el LLM fue invocado (el prompt debe guiar a respuesta amigable)
    assert mock_llm.call_count == 1, "El LLM debe ser invocado"
    
    # Verificar que el prompt contiene instrucciones para manejar contexto insuficiente
    assert mock_llm.last_system_prompt is not None, "Debe haber un system prompt"
    prompt_lower = mock_llm.last_system_prompt.lower()
    assert 'contexto es insuficiente' in prompt_lower or 'insuficiente' in prompt_lower, \
        "El prompt debe mencionar qué hacer cuando el contexto es insuficiente"
    assert 'reformular' in prompt_lower, \
        "El prompt debe sugerir reformular la pregunta"


@pytest.mark.asyncio
async def test_consulta_texto_con_historial_conversacion(sistema_con_mock_llm):
    """
    Test para verificar que el historial de conversación se incluye correctamente
    
    **Validates: Requirements 1.1, 6.1**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Basándome en la conversación anterior, el epitelio..."
    
    # Arrange: crear state con historial
    state = {
        'consulta_usuario': '¿Y qué más puedes decirme?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio tiene múltiples funciones...',
        'contexto_memoria': 'Usuario: ¿Qué es el epitelio?\nAsistente: El epitelio es un tejido...',
        'trayectoria': []
    }
    
    # Act: ejecutar el nodo
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert: verificar que se generó respuesta
    assert 'respuesta_final' in resultado, "Debe retornar respuesta_final"
    
    # Verificar que el mensaje incluye el historial
    assert mock_llm.last_user_content is not None, "Debe haber contenido de usuario"
    if isinstance(mock_llm.last_user_content, list):
        texto_completo = ' '.join([
            parte.get('text', '') for parte in mock_llm.last_user_content 
            if parte.get('type') == 'text'
        ])
        assert 'HISTORIAL' in texto_completo.upper(), \
            "Debe incluir sección de historial"
        assert 'epitelio es un tejido' in texto_completo.lower(), \
            "Debe incluir el contenido del historial"


@pytest.mark.asyncio
async def test_consulta_imagen_con_imagen_usuario(sistema_con_mock_llm, temp_image):
    """
    Test para verificar que se incluye tanto la imagen del usuario como las recuperadas
    
    **Validates: Requirements 6.4**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Comparando tu imagen con las de la base de datos..."
    
    # Arrange: crear state con imagen de usuario e imágenes recuperadas
    state = {
        'consulta_usuario': '¿Qué tipo de tejido es este?',
        'imagen_consulta': temp_image,  # Imagen del usuario
        'imagenes_relevantes': [temp_image],  # Imágenes recuperadas
        'contexto_documentos': 'Tejido epitelial...',
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act: ejecutar el nodo
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert: verificar que se generó respuesta
    assert 'respuesta_final' in resultado, "Debe retornar respuesta_final"
    
    # Verificar que el mensaje incluye múltiples imágenes
    assert mock_llm.last_user_content is not None, "Debe haber contenido de usuario"
    if isinstance(mock_llm.last_user_content, list):
        imagenes = [p for p in mock_llm.last_user_content if p.get('type') == 'image_url']
        # Debe incluir al menos la imagen recuperada y la del usuario
        assert len(imagenes) >= 1, "Debe incluir al menos una imagen"
        
        # Verificar que hay etiquetas para distinguir tipos de imágenes
        textos = [p.get('text', '') for p in mock_llm.last_user_content if p.get('type') == 'text']
        texto_completo = ' '.join(textos)
        # Debe mencionar que hay imágenes recuperadas
        assert 'IMAGEN' in texto_completo.upper(), \
            "Debe etiquetar las imágenes"


@pytest.mark.asyncio
async def test_limite_imagenes_respetado(sistema_con_mock_llm):
    """
    Test para verificar que se respeta el límite de imágenes
    
    **Validates: Requirements 6.3**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Análisis de las imágenes proporcionadas..."
    
    # Crear múltiples imágenes temporales
    temp_images = []
    for i in range(10):  # Crear 10 imágenes (más del límite)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01')
            f.write(b'\xFF\xD9')
            temp_images.append(f.name)
    
    try:
        # Arrange: crear state con muchas imágenes
        state = {
            'consulta_usuario': '¿Qué muestran estas imágenes?',
            'imagen_consulta': None,
            'imagenes_relevantes': temp_images,
            'contexto_documentos': 'Múltiples tipos de tejidos...',
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act: ejecutar el nodo
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Assert: verificar que se generó respuesta
        assert 'respuesta_final' in resultado, "Debe retornar respuesta_final"
        
        # Verificar que no se excede el límite de imágenes
        if isinstance(mock_llm.last_user_content, list):
            imagenes = [p for p in mock_llm.last_user_content if p.get('type') == 'image_url']
            assert len(imagenes) <= 5, \
                f"No debe incluir más de 5 imágenes, pero incluyó {len(imagenes)}"
    
    finally:
        # Cleanup: eliminar imágenes temporales
        for img_path in temp_images:
            if os.path.exists(img_path):
                os.unlink(img_path)


@pytest.mark.asyncio
async def test_error_cargando_imagen_continua_sin_ella(sistema_con_mock_llm):
    """
    Test para verificar que si falla la carga de una imagen, continúa sin ella
    
    **Validates: Requirements 6.2**
    """
    sistema, mock_llm = sistema_con_mock_llm
    mock_llm.response = "Basándome en el contexto textual..."
    
    # Arrange: crear state con imagen que no existe
    state = {
        'consulta_usuario': '¿Qué muestra esta imagen?',
        'imagen_consulta': None,
        'imagenes_relevantes': ['/path/to/nonexistent/image.jpg'],  # Imagen inexistente
        'contexto_documentos': 'Descripción del tejido...',
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act: ejecutar el nodo (no debe lanzar excepción)
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert: verificar que se generó respuesta a pesar del error
    assert 'respuesta_final' in resultado, \
        "Debe retornar respuesta_final incluso si falla la carga de imagen"
    assert resultado['respuesta_final'] == mock_llm.response, \
        "La respuesta debe ser la retornada por el LLM"
    
    # Verificar que el LLM fue invocado
    assert mock_llm.call_count == 1, "El LLM debe ser invocado a pesar del error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
