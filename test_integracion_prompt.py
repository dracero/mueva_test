"""
Test de integración para verificar que _generar_prompt_sistema se integra correctamente
"""
from muvera_test import SistemaRAGColPaliPuro


def test_integracion_detectar_y_generar_prompt():
    """Verificar que _detectar_tipo_consulta y _generar_prompt_sistema funcionan juntos"""
    sistema = SistemaRAGColPaliPuro()
    
    # Test 1: Consulta de solo texto
    state_texto = {
        'imagen_consulta': None,
        'imagenes_relevantes': []
    }
    
    tipo = sistema._detectar_tipo_consulta(state_texto)
    assert tipo == 'texto', f"Esperaba 'texto', obtuvo '{tipo}'"
    
    prompt = sistema._generar_prompt_sistema(tipo)
    assert len(prompt) > 0, "El prompt no debe estar vacío"
    assert 'imagen recuperada' not in prompt.lower(), "Prompt de texto no debe mencionar imágenes recuperadas"
    
    print("✅ Test integración: consulta de texto funciona correctamente")
    
    # Test 2: Consulta con imagen
    state_imagen = {
        'imagen_consulta': None,
        'imagenes_relevantes': ['imagen1.jpg', 'imagen2.jpg']
    }
    
    tipo = sistema._detectar_tipo_consulta(state_imagen)
    assert tipo == 'imagen', f"Esperaba 'imagen', obtuvo '{tipo}'"
    
    prompt = sistema._generar_prompt_sistema(tipo)
    assert len(prompt) > 0, "El prompt no debe estar vacío"
    assert 'imagen recuperada' in prompt.lower(), "Prompt de imagen debe mencionar imágenes recuperadas"
    
    print("✅ Test integración: consulta con imagen funciona correctamente")


def test_prompts_diferentes_segun_tipo():
    """Verificar que los prompts son diferentes según el tipo de consulta"""
    sistema = SistemaRAGColPaliPuro()
    
    prompt_texto = sistema._generar_prompt_sistema('texto')
    prompt_imagen = sistema._generar_prompt_sistema('imagen')
    
    # Los prompts deben ser diferentes
    assert prompt_texto != prompt_imagen, "Los prompts deben ser diferentes"
    
    # Ambos deben tener contenido sustancial
    assert len(prompt_texto) > 200, "Prompt de texto debe tener contenido sustancial"
    assert len(prompt_imagen) > 200, "Prompt de imagen debe tener contenido sustancial"
    
    # Verificar diferencias clave
    assert 'imagen recuperada' not in prompt_texto.lower(), "Prompt texto no debe mencionar imágenes"
    assert 'imagen recuperada' in prompt_imagen.lower(), "Prompt imagen debe mencionar imágenes"
    
    print("✅ Test integración: prompts son diferentes según tipo de consulta")


if __name__ == "__main__":
    print("\n🧪 Ejecutando tests de integración...\n")
    
    try:
        test_integracion_detectar_y_generar_prompt()
        test_prompts_diferentes_segun_tipo()
        
        print("\n✅ Todos los tests de integración pasaron!")
        
    except AssertionError as e:
        print(f"\n❌ Test falló: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import sys
        sys.exit(1)
