"""
Tests unitarios para el método _detectar_tipo_consulta
"""
import os
import sys
import tempfile
import pytest

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from muvera_test import SistemaRAGColPaliPuro


@pytest.fixture
def sistema():
    """Fixture que crea una instancia del sistema para testing"""
    return SistemaRAGColPaliPuro()


@pytest.fixture
def temp_image():
    """Fixture que crea una imagen temporal para testing"""
    # Crear un archivo temporal que simula una imagen
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
        f.write("fake image content")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_detectar_tipo_consulta_con_imagen_usuario_valida(sistema, temp_image):
    """Debe detectar 'imagen' cuando hay imagen_consulta válida"""
    state = {
        'imagen_consulta': temp_image,
        'imagenes_relevantes': []
    }
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'imagen', "Debe detectar 'imagen' cuando hay imagen_consulta válida"


def test_detectar_tipo_consulta_con_imagenes_recuperadas(sistema):
    """Debe detectar 'imagen' cuando hay imagenes_relevantes"""
    state = {
        'imagen_consulta': None,
        'imagenes_relevantes': ['/path/to/retrieved.jpg']
    }
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'imagen', "Debe detectar 'imagen' cuando hay imagenes_relevantes"


def test_detectar_tipo_consulta_solo_texto(sistema):
    """Debe detectar 'texto' cuando no hay imágenes"""
    state = {
        'imagen_consulta': None,
        'imagenes_relevantes': []
    }
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'texto', "Debe detectar 'texto' cuando no hay imágenes"


def test_detectar_tipo_consulta_imagen_invalida(sistema):
    """Debe detectar 'texto' si imagen_consulta no existe en filesystem"""
    state = {
        'imagen_consulta': '/path/nonexistent.jpg',
        'imagenes_relevantes': []
    }
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'texto', "Debe detectar 'texto' cuando imagen_consulta no existe"


def test_detectar_tipo_consulta_con_ambas_imagenes(sistema, temp_image):
    """Debe detectar 'imagen' cuando hay tanto imagen_consulta como imagenes_relevantes"""
    state = {
        'imagen_consulta': temp_image,
        'imagenes_relevantes': ['/path/to/retrieved.jpg']
    }
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'imagen', "Debe detectar 'imagen' cuando hay ambos tipos de imágenes"


def test_detectar_tipo_consulta_imagenes_relevantes_vacia(sistema):
    """Debe detectar 'texto' cuando imagenes_relevantes es una lista vacía"""
    state = {
        'imagen_consulta': None,
        'imagenes_relevantes': []
    }
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'texto', "Debe detectar 'texto' cuando imagenes_relevantes está vacía"


def test_detectar_tipo_consulta_sin_claves(sistema):
    """Debe detectar 'texto' cuando el state no tiene las claves esperadas"""
    state = {}
    resultado = sistema._detectar_tipo_consulta(state)
    assert resultado == 'texto', "Debe detectar 'texto' cuando no hay claves en el state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
