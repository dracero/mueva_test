"""
Tests para validación de contexto insuficiente (Task 6.1)

**Validates: Requirements 5.1, 5.3**

Este módulo verifica que el sistema valide la longitud mínima del contexto
y guíe apropiadamente al LLM cuando el contexto es insuficiente.
"""

import pytest
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
async def test_contexto_vacio_guia_llm_apropiadamente(sistema_con_mock_llm):
    """
    Verifica que cuando el contexto está vacío, el prompt guía al LLM apropiadamente.
    
    **Validates: Requirements 5.1, 5.3**
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    
    # State con contexto vacío
    state = {
        'consulta_usuario': '¿Qué es el tejido conectivo?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': '',  # Vacío
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert
    assert mock_llm.call_count == 1
    assert mock_llm.last_system_prompt is not None
    
    # El prompt debe contener instrucciones sobre qué hacer con contexto insuficiente
    prompt_lower = mock_llm.last_system_prompt.lower()
    assert 'insuficiente' in prompt_lower or 'no tengo suficiente' in prompt_lower
    assert 'reformular' in prompt_lower


@pytest.mark.asyncio
async def test_contexto_muy_corto_guia_llm_apropiadamente(sistema_con_mock_llm):
    """
    Verifica que cuando el contexto es muy corto (<50 caracteres), 
    el prompt guía al LLM apropiadamente.
    
    **Validates: Requirements 5.1, 5.3**
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    
    # State con contexto muy corto (menos de 50 caracteres)
    state = {
        'consulta_usuario': '¿Qué es el tejido conectivo?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Texto muy corto.',  # Solo 17 caracteres
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert
    assert mock_llm.call_count == 1
    assert mock_llm.last_system_prompt is not None
    
    # El prompt debe contener instrucciones sobre qué hacer con contexto insuficiente
    prompt_lower = mock_llm.last_system_prompt.lower()
    assert 'insuficiente' in prompt_lower or 'no tengo suficiente' in prompt_lower


@pytest.mark.asyncio
async def test_contexto_suficiente_no_agrega_advertencia(sistema_con_mock_llm):
    """
    Verifica que cuando el contexto es suficiente (>50 caracteres),
    el sistema funciona normalmente sin advertencias adicionales.
    
    **Validates: Requirements 5.1**
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    mock_llm.response = "El tejido conectivo es un tipo de tejido que proporciona soporte estructural..."
    
    # State con contexto suficiente
    state = {
        'consulta_usuario': '¿Qué es el tejido conectivo?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El tejido conectivo es un tipo de tejido que proporciona soporte estructural y metabólico a otros tejidos del cuerpo.',  # >50 caracteres
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert
    assert mock_llm.call_count == 1
    assert resultado['respuesta_final'] == mock_llm.response
    
    # El sistema debe funcionar normalmente
    assert 'respuesta_final' in resultado


@pytest.mark.asyncio
async def test_validacion_contexto_con_espacios_en_blanco(sistema_con_mock_llm):
    """
    Verifica que la validación de contexto ignore espacios en blanco.
    
    **Validates: Requirements 5.1**
    """
    sistema = sistema_con_mock_llm
    mock_llm = sistema.llm
    
    # State con contexto que solo tiene espacios
    state = {
        'consulta_usuario': '¿Qué es el tejido conectivo?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': '     \n\n   \t   ',  # Solo espacios en blanco
        'contexto_memoria': '',
        'trayectoria': []
    }
    
    # Act
    resultado = await sistema._nodo_generar_respuesta(state)
    
    # Assert
    assert mock_llm.call_count == 1
    
    # El prompt debe guiar al LLM sobre contexto insuficiente
    prompt_lower = mock_llm.last_system_prompt.lower()
    assert 'insuficiente' in prompt_lower or 'no tengo suficiente' in prompt_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
