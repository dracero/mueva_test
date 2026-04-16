"""
Tests unitarios para el método _construir_mensaje_usuario

**Validates: Requirements 1.2, 4.1, 4.2, 6.3**
"""
import os
import sys
import pytest

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from muvera_test import SistemaRAGColPaliPuro


@pytest.fixture
def sistema():
    """Fixture que crea una instancia del sistema para testing"""
    return SistemaRAGColPaliPuro()


def test_construir_mensaje_texto_solo_contiene_texto(sistema):
    """Mensaje de texto no debe incluir partes de imagen
    
    **Validates: Requirements 1.2, 4.1**
    """
    state = {
        'consulta_usuario': '¿Qué es el epitelio?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'El epitelio es un tejido que recubre superficies.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    
    tipos = [parte['type'] for parte in mensaje]
    assert 'text' in tipos, "Debe contener partes de texto"
    assert 'image_url' not in tipos, "No debe contener partes de imagen"


def test_construir_mensaje_imagen_incluye_imagenes(sistema):
    """Mensaje de imagen debe incluir partes de imagen cuando hay imágenes disponibles
    
    **Validates: Requirements 4.2, 6.3**
    """
    # Usar una imagen real del proyecto
    imagen_test = 'histopatologia_data/embeddings/arch2_p1_img1.jpg'
    
    if not os.path.exists(imagen_test):
        pytest.skip(f"Imagen de prueba no encontrada: {imagen_test}")
    
    state = {
        'consulta_usuario': '¿Qué muestra esta imagen?',
        'imagen_consulta': None,
        'imagenes_relevantes': [imagen_test],
        'contexto_documentos': 'Contexto sobre la imagen.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    
    tipos = [parte['type'] for parte in mensaje]
    assert 'text' in tipos, "Debe contener partes de texto"
    assert 'image_url' in tipos, "Debe contener partes de imagen"


def test_construir_mensaje_respeta_limite_imagenes(sistema):
    """No debe incluir más de max_imagenes
    
    **Validates: Requirements 6.3**
    """
    # Crear lista de 10 imágenes (más del límite)
    imagenes_disponibles = [
        f'histopatologia_data/embeddings/arch2_p{i}_img{i}.jpg' 
        for i in range(1, 11)
    ]
    
    # Filtrar solo las que existen
    imagenes_existentes = [img for img in imagenes_disponibles if os.path.exists(img)]
    
    if len(imagenes_existentes) < 6:
        pytest.skip(f"No hay suficientes imágenes de prueba (encontradas: {len(imagenes_existentes)})")
    
    state = {
        'consulta_usuario': '¿Qué muestran estas imágenes?',
        'imagen_consulta': None,
        'imagenes_relevantes': imagenes_existentes,
        'contexto_documentos': 'Contexto sobre las imágenes.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    
    imagenes = [p for p in mensaje if p['type'] == 'image_url']
    assert len(imagenes) <= 5, f"No debe incluir más de 5 imágenes, pero incluyó {len(imagenes)}"


def test_construir_mensaje_incluye_contexto(sistema):
    """Mensaje debe incluir contexto_documentos
    
    **Validates: Requirements 4.1**
    """
    contexto_prueba = "Este es un contexto de prueba sobre histopatología"
    
    state = {
        'consulta_usuario': '¿Qué es el epitelio?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': contexto_prueba,
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    assert contexto_prueba in texto_completo, "Debe incluir el contexto de documentos"


def test_construir_mensaje_incluye_historial(sistema):
    """Mensaje debe incluir historial de conversación si existe
    
    **Validates: Requirements 4.1**
    """
    historial_prueba = "Usuario preguntó antes sobre tejidos"
    
    state = {
        'consulta_usuario': '¿Y qué más?',
        'imagen_consulta': None,
        'imagenes_relevantes': [],
        'contexto_documentos': 'Contexto actual',
        'contexto_memoria': historial_prueba
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'texto')
    
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    assert historial_prueba in texto_completo, "Debe incluir el historial de conversación"


def test_construir_mensaje_todas_imagenes_fallan(sistema):
    """Si todas las imágenes fallan al cargar, debe tratar como consulta de texto
    
    **Validates: Requirements 6.2**
    """
    # Usar paths de imágenes que no existen
    imagenes_inexistentes = [
        'path/inexistente/imagen1.jpg',
        'path/inexistente/imagen2.jpg',
        'path/inexistente/imagen3.jpg'
    ]
    
    state = {
        'consulta_usuario': '¿Qué muestra esta imagen?',
        'imagen_consulta': None,
        'imagenes_relevantes': imagenes_inexistentes,
        'contexto_documentos': 'Contexto sobre histopatología.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    
    # Debe comportarse como consulta de texto
    tipos = [parte['type'] for parte in mensaje]
    assert 'text' in tipos, "Debe contener partes de texto"
    assert 'image_url' not in tipos, "No debe contener partes de imagen si todas fallaron"
    
    # Verificar que el mensaje no menciona imágenes adjuntas
    texto_completo = ' '.join([p['text'] for p in mensaje if p['type'] == 'text'])
    assert 'IMÁGENES adjuntas' not in texto_completo, "No debe mencionar imágenes adjuntas"
    assert 'contexto de arriba' in texto_completo, "Debe usar formato de texto"


def test_construir_mensaje_algunas_imagenes_fallan(sistema):
    """Si algunas imágenes fallan pero otras cargan, debe incluir las exitosas
    
    **Validates: Requirements 6.2**
    """
    # Mezclar imágenes existentes e inexistentes
    imagen_existente = 'histopatologia_data/embeddings/arch2_p1_img1.jpg'
    
    if not os.path.exists(imagen_existente):
        pytest.skip(f"Imagen de prueba no encontrada: {imagen_existente}")
    
    imagenes_mixtas = [
        'path/inexistente/imagen1.jpg',
        imagen_existente,
        'path/inexistente/imagen2.jpg'
    ]
    
    state = {
        'consulta_usuario': '¿Qué muestra esta imagen?',
        'imagen_consulta': None,
        'imagenes_relevantes': imagenes_mixtas,
        'contexto_documentos': 'Contexto sobre histopatología.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    
    # Debe incluir al menos una imagen
    tipos = [parte['type'] for parte in mensaje]
    assert 'text' in tipos, "Debe contener partes de texto"
    assert 'image_url' in tipos, "Debe contener al menos una imagen exitosa"
    
    # Contar cuántas imágenes se cargaron
    imagenes = [p for p in mensaje if p['type'] == 'image_url']
    assert len(imagenes) == 1, f"Debe cargar exactamente 1 imagen exitosa, pero cargó {len(imagenes)}"


def test_construir_mensaje_imagen_consulta_falla(sistema):
    """Si la imagen de consulta del usuario falla, debe continuar sin ella
    
    **Validates: Requirements 6.2**
    """
    imagen_recuperada = 'histopatologia_data/embeddings/arch2_p1_img1.jpg'
    
    if not os.path.exists(imagen_recuperada):
        pytest.skip(f"Imagen de prueba no encontrada: {imagen_recuperada}")
    
    state = {
        'consulta_usuario': '¿Qué muestra esta imagen?',
        'imagen_consulta': 'path/inexistente/query_image.jpg',  # No existe
        'imagenes_relevantes': [imagen_recuperada],
        'contexto_documentos': 'Contexto sobre histopatología.',
        'contexto_memoria': ''
    }
    
    mensaje = sistema._construir_mensaje_usuario(state, 'imagen')
    
    # Debe incluir la imagen recuperada pero no la de consulta
    tipos = [parte['type'] for parte in mensaje]
    assert 'image_url' in tipos, "Debe incluir la imagen recuperada"
    
    # Contar imágenes - solo debe haber 1 (la recuperada)
    imagenes = [p for p in mensaje if p['type'] == 'image_url']
    assert len(imagenes) == 1, f"Debe cargar solo la imagen recuperada, pero cargó {len(imagenes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
