"""
Verificación de que la tarea 2.1 cumple con todos los requisitos especificados
"""
from muvera_test import SistemaRAGColPaliPuro


def verificar_requisito_2_1():
    """Requirement 2.1: Sistema debe usar tono conversacional"""
    sistema = SistemaRAGColPaliPuro()
    
    for tipo in ['texto', 'imagen']:
        prompt = sistema._generar_prompt_sistema(tipo)
        
        # Verificar tono conversacional
        assert 'profesor' in prompt.lower() and 'amigable' in prompt.lower(), \
            f"Req 2.1: Prompt {tipo} debe usar tono conversacional (profesor amigable)"
    
    print("✅ Requisito 2.1: Tono conversacional implementado")


def verificar_requisito_2_2():
    """Requirement 2.2: Responder como profesor amigable"""
    sistema = SistemaRAGColPaliPuro()
    
    for tipo in ['texto', 'imagen']:
        prompt = sistema._generar_prompt_sistema(tipo)
        
        # Verificar que instruye al modelo a actuar como profesor
        assert 'profesor experto' in prompt.lower(), \
            f"Req 2.2: Prompt {tipo} debe instruir a responder como profesor"
        assert 'estudiantes' in prompt.lower() or 'ayudar' in prompt.lower(), \
            f"Req 2.2: Prompt {tipo} debe mencionar ayudar a estudiantes"
    
    print("✅ Requisito 2.2: Instrucciones de profesor amigable implementadas")


def verificar_requisito_2_3():
    """Requirement 2.3: Evitar lenguaje excesivamente técnico"""
    sistema = SistemaRAGColPaliPuro()
    
    for tipo in ['texto', 'imagen']:
        prompt = sistema._generar_prompt_sistema(tipo)
        
        # Verificar que menciona claridad y accesibilidad
        assert 'clara' in prompt.lower() or 'accesible' in prompt.lower(), \
            f"Req 2.3: Prompt {tipo} debe mencionar claridad/accesibilidad"
    
    print("✅ Requisito 2.3: Instrucciones para evitar lenguaje técnico excesivo")


def verificar_requisito_2_4():
    """Requirement 2.4: Mantener precisión y rigor científico"""
    sistema = SistemaRAGColPaliPuro()
    
    for tipo in ['texto', 'imagen']:
        prompt = sistema._generar_prompt_sistema(tipo)
        
        # Verificar que mantiene rigor científico
        assert 'rigor científico' in prompt.lower() or 'precisión' in prompt.lower(), \
            f"Req 2.4: Prompt {tipo} debe mantener rigor científico"
        assert 'contexto' in prompt.lower(), \
            f"Req 2.4: Prompt {tipo} debe basarse en contexto"
    
    print("✅ Requisito 2.4: Rigor científico mantenido")


def verificar_requisito_3_1():
    """Requirement 3.1: Prompt enfocado en análisis textual para consultas de texto"""
    sistema = SistemaRAGColPaliPuro()
    
    prompt_texto = sistema._generar_prompt_sistema('texto')
    
    # Verificar que se enfoca en análisis textual
    assert 'contexto textual' in prompt_texto.lower() or 'contexto' in prompt_texto.lower(), \
        "Req 3.1: Prompt texto debe enfocarse en análisis textual"
    
    # Verificar que NO menciona imágenes
    assert 'imagen recuperada' not in prompt_texto.lower(), \
        "Req 3.1: Prompt texto no debe mencionar imágenes recuperadas"
    
    print("✅ Requisito 3.1: Prompt de texto enfocado en análisis textual")


def verificar_requisito_3_2():
    """Requirement 3.2: Prompt con instrucciones visuales para consultas con imagen"""
    sistema = SistemaRAGColPaliPuro()
    
    prompt_imagen = sistema._generar_prompt_sistema('imagen')
    
    # Verificar que incluye instrucciones visuales
    assert 'imagen recuperada' in prompt_imagen.lower(), \
        "Req 3.2: Prompt imagen debe mencionar imágenes recuperadas"
    assert 'análisis visual' in prompt_imagen.lower() or 'analiza' in prompt_imagen.lower(), \
        "Req 3.2: Prompt imagen debe incluir instrucciones de análisis visual"
    
    print("✅ Requisito 3.2: Prompt de imagen incluye instrucciones visuales")


def verificar_requisito_3_4():
    """Requirement 3.4: Omitir instrucciones de imágenes cuando no hay contexto visual"""
    sistema = SistemaRAGColPaliPuro()
    
    prompt_texto = sistema._generar_prompt_sistema('texto')
    
    # Verificar que omite instrucciones de análisis visual
    assert 'análisis visual' not in prompt_texto.lower(), \
        "Req 3.4: Prompt texto debe omitir instrucciones de análisis visual"
    assert 'imagen encontrada' not in prompt_texto.lower(), \
        "Req 3.4: Prompt texto debe omitir sección de imagen encontrada"
    
    print("✅ Requisito 3.4: Instrucciones de imágenes omitidas en prompt de texto")


def verificar_requisito_5_2():
    """Requirement 5.2: Reemplazar mensaje técnico con mensaje conversacional"""
    sistema = SistemaRAGColPaliPuro()
    
    for tipo in ['texto', 'imagen']:
        prompt = sistema._generar_prompt_sistema(tipo)
        
        # Verificar que incluye mensaje amigable para contexto insuficiente
        assert 'no tengo suficiente información' in prompt.lower(), \
            f"Req 5.2: Prompt {tipo} debe incluir mensaje amigable para contexto insuficiente"
        
        # Verificar que es conversacional (no técnico)
        assert 'reformular' in prompt.lower() or 'más detalles' in prompt.lower(), \
            f"Req 5.2: Prompt {tipo} debe sugerir reformular de forma amigable"
    
    print("✅ Requisito 5.2: Mensaje de error conversacional implementado")


def verificar_requisito_5_3():
    """Requirement 5.3: Sugerir reformular cuando contexto es insuficiente"""
    sistema = SistemaRAGColPaliPuro()
    
    for tipo in ['texto', 'imagen']:
        prompt = sistema._generar_prompt_sistema(tipo)
        
        # Verificar que sugiere reformular
        assert 'reformular' in prompt.lower() or 'más detalles' in prompt.lower(), \
            f"Req 5.3: Prompt {tipo} debe sugerir reformular la pregunta"
    
    print("✅ Requisito 5.3: Sugerencia de reformular implementada")


if __name__ == "__main__":
    print("\n🔍 Verificando cumplimiento de requisitos de la tarea 2.1...\n")
    
    try:
        verificar_requisito_2_1()
        verificar_requisito_2_2()
        verificar_requisito_2_3()
        verificar_requisito_2_4()
        verificar_requisito_3_1()
        verificar_requisito_3_2()
        verificar_requisito_3_4()
        verificar_requisito_5_2()
        verificar_requisito_5_3()
        
        print("\n✅ Todos los requisitos de la tarea 2.1 se cumplen correctamente!")
        print("\nRequisitos verificados:")
        print("  - 2.1: Tono conversacional")
        print("  - 2.2: Responder como profesor amigable")
        print("  - 2.3: Evitar lenguaje técnico excesivo")
        print("  - 2.4: Mantener rigor científico")
        print("  - 3.1: Prompt enfocado en análisis textual")
        print("  - 3.2: Prompt con instrucciones visuales")
        print("  - 3.4: Omitir instrucciones de imágenes en texto")
        print("  - 5.2: Mensaje de error conversacional")
        print("  - 5.3: Sugerir reformular pregunta")
        
    except AssertionError as e:
        print(f"\n❌ Requisito no cumplido: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import sys
        sys.exit(1)
