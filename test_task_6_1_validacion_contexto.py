"""
Test completo para Task 6.1: Validación de contexto insuficiente

**Validates: Requirements 5.1, 5.3**

Este test verifica que:
1. Se valida la longitud mínima de contexto_documentos (>50 caracteres)
2. El prompt guía al LLM a responder con mensaje amigable cuando el contexto es insuficiente
3. El sistema registra un log cuando detecta contexto insuficiente
"""

import pytest
import io
import sys
from muvera_test import SistemaRAGColPaliPuro


class MockLLM:
    """Mock del LLM para testing"""
    def __init__(self, response="Respuesta de prueba"):
        self.response = response
        self.call_count = 0
        self.last_messages = None
        self.last_system_prompt = None
        self.last_user_content = None
    
    async def ainvoke(self, messages):
        self.call_count += 1
        self.last_messages = messages
        
        # Extraer system prompt y user content
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'system':
                    self.last_system_prompt = msg.content
                elif msg.type == 'human':
                    self.last_user_content = msg.content
        
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(self.response)


@pytest.fixture
def sistema_con_mock_llm():
    """Sistema con LLM mockeado"""
    sistema = SistemaRAGColPaliPuro()
    sistema.llm = MockLLM(
        response="No tengo suficiente información en mis fuentes para responder eso con precisión. ¿Podrías reformular tu pregunta o darme más detalles?"
    )
    return sistema


@pytest.mark.asyncio
async def test_task_6_1_validacion_contexto_vacio(sistema_con_mock_llm):
    """
    Task 6.1: Verifica validación cuando contexto está vacío.
    
    **Validates: Requirements 5.1, 5.3**
    
    Verifica que:
    - Se detecta contexto insuficiente (0 caracteres < 50)
    - Se registra un log de advertencia
    - El prompt guía al LLM apropiadamente
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    
    # Capturar output para verificar el log
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # State con contexto vacío
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': '',  # 0 caracteres
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Restaurar stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert 1: Se registró el log de contexto insuficiente
        assert '⚠️ Contexto insuficiente detectado' in output, \
            "Debe registrar advertencia cuando contexto es insuficiente"
        
        # Assert 2: El LLM fue invocado
        assert mock_llm.call_count == 1
        
        # Assert 3: El prompt contiene instrucciones para contexto insuficiente
        prompt_lower = mock_llm.last_system_prompt.lower()
        assert 'insuficiente' in prompt_lower or 'no tengo suficiente' in prompt_lower, \
            "El prompt debe mencionar qué hacer cuando el contexto es insuficiente"
        assert 'reformular' in prompt_lower, \
            "El prompt debe sugerir reformular la pregunta"
        
        # Assert 4: Se generó una respuesta
        assert 'respuesta_final' in resultado
        assert resultado['respuesta_final'] == mock_llm.response
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_task_6_1_validacion_contexto_muy_corto(sistema_con_mock_llm):
    """
    Task 6.1: Verifica validación cuando contexto es muy corto (<50 caracteres).
    
    **Validates: Requirements 5.1, 5.3**
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    
    # Capturar output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # State con contexto de 20 caracteres (< 50)
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': 'Texto muy corto.',  # 17 caracteres
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Restaurar stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Se registró el log de contexto insuficiente
        assert '⚠️ Contexto insuficiente detectado' in output
        
        # Assert: El prompt guía al LLM apropiadamente
        prompt_lower = mock_llm.last_system_prompt.lower()
        assert 'insuficiente' in prompt_lower or 'no tengo suficiente' in prompt_lower
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_task_6_1_validacion_contexto_solo_espacios(sistema_con_mock_llm):
    """
    Task 6.1: Verifica que la validación use strip() para ignorar espacios.
    
    **Validates: Requirements 5.1**
    """
    sistema = sistema_con_mock_llm
    
    # Capturar output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # State con contexto que solo tiene espacios en blanco
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': '     \n\n   \t   ',  # Solo espacios
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Restaurar stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Se detectó como contexto insuficiente (strip() elimina espacios)
        assert '⚠️ Contexto insuficiente detectado' in output
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_task_6_1_contexto_suficiente_no_registra_advertencia(sistema_con_mock_llm):
    """
    Task 6.1: Verifica que NO se registra advertencia cuando el contexto es suficiente.
    
    **Validates: Requirements 5.1**
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    mock_llm.response = "El tejido conectivo es un tipo de tejido..."
    
    # Capturar output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # State con contexto suficiente (>50 caracteres)
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': 'El tejido conectivo es un tipo de tejido que proporciona soporte estructural y metabólico a otros tejidos del cuerpo.',  # 120 caracteres
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Restaurar stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: NO se registró advertencia de contexto insuficiente
        assert '⚠️ Contexto insuficiente detectado' not in output, \
            "No debe registrar advertencia cuando el contexto es suficiente"
        
        # Assert: El sistema funcionó normalmente
        assert resultado['respuesta_final'] == mock_llm.response
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_task_6_1_umbral_exacto_50_caracteres(sistema_con_mock_llm):
    """
    Task 6.1: Verifica el comportamiento en el umbral exacto de 50 caracteres.
    
    **Validates: Requirements 5.1**
    """
    sistema = sistema_con_mock_llm
    
    # Capturar output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # State con exactamente 50 caracteres (debe ser suficiente)
        contexto_50 = 'A' * 50  # Exactamente 50 caracteres
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': contexto_50,
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Restaurar stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: NO se registró advertencia (50 caracteres es suficiente)
        assert '⚠️ Contexto insuficiente detectado' not in output, \
            "50 caracteres debe considerarse suficiente (umbral es <50)"
        
    finally:
        sys.stdout = sys.__stdout__


@pytest.mark.asyncio
async def test_task_6_1_umbral_49_caracteres(sistema_con_mock_llm):
    """
    Task 6.1: Verifica que 49 caracteres se considera insuficiente.
    
    **Validates: Requirements 5.1**
    """
    sistema = sistema_con_mock_llm
    
    # Capturar output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # State con 49 caracteres (debe ser insuficiente)
        contexto_49 = 'A' * 49  # 49 caracteres
        state = {
            'consulta_usuario': '¿Qué es el tejido conectivo?',
            'imagen_consulta': None,
            'imagenes_relevantes': [],
            'contexto_documentos': contexto_49,
            'contexto_memoria': '',
            'trayectoria': []
        }
        
        # Act
        resultado = await sistema._nodo_generar_respuesta(state)
        
        # Restaurar stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Assert: Se registró advertencia (49 < 50)
        assert '⚠️ Contexto insuficiente detectado' in output, \
            "49 caracteres debe considerarse insuficiente (umbral es <50)"
        
    finally:
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
