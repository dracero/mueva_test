"""
Tests unitarios para el método _generar_prompt_sistema

Requisitos validados:
- 2.1: Tono conversacional en respuestas
- 2.3: Mantener precisión y rigor científico
- 2.4: Mantener precisión y rigor científico mientras usa tono conversacional
- 3.1: Prompt enfocado en análisis textual para consultas de texto
- 3.2: Prompt con instrucciones para análisis visual para consultas con imagen
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


class TestPromptTextoElementosConversacionales:
    """Test: prompt de texto contiene elementos conversacionales y no menciona imágenes"""
    
    def test_prompt_texto_contiene_profesor_experto(self, sistema):
        """Debe contener 'profesor experto' para establecer tono educativo"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert 'profesor experto' in prompt.lower(), \
            "El prompt debe establecer el rol de profesor experto"
    
    def test_prompt_texto_contiene_amigable(self, sistema):
        """Debe contener 'amigable' para establecer tono conversacional"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert 'amigable' in prompt.lower(), \
            "El prompt debe mencionar estilo amigable"
    
    def test_prompt_texto_contiene_educativo(self, sistema):
        """Debe contener 'educativo' para establecer propósito pedagógico"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert 'educativo' in prompt.lower(), \
            "El prompt debe mencionar estilo educativo"
    
    def test_prompt_texto_menciona_contexto_textual(self, sistema):
        """Debe mencionar 'contexto textual' para indicar fuente de información"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert 'contexto textual' in prompt.lower(), \
            "El prompt debe mencionar que se basa en contexto textual"
    
    def test_prompt_texto_no_menciona_imagenes(self, sistema):
        """No debe mencionar 'imagen' ya que es para consultas de solo texto"""
        prompt = sistema._generar_prompt_sistema('texto')
        # Verificar que no menciona imagen/imágenes/visual/análisis visual
        prompt_lower = prompt.lower()
        assert 'imagen' not in prompt_lower, \
            "El prompt de texto no debe mencionar imágenes"
        assert 'visual' not in prompt_lower, \
            "El prompt de texto no debe mencionar análisis visual"
    
    def test_prompt_texto_contiene_tono_conversacional(self, sistema):
        """Debe mencionar explícitamente 'tono conversacional'"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert 'tono conversacional' in prompt.lower(), \
            "El prompt debe instruir usar tono conversacional"


class TestPromptImagenElementosVisuales:
    """Test: prompt de imagen contiene instrucciones visuales y tono conversacional"""
    
    def test_prompt_imagen_contiene_profesor_experto(self, sistema):
        """Debe contener 'profesor experto' para establecer tono educativo"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert 'profesor experto' in prompt.lower(), \
            "El prompt debe establecer el rol de profesor experto"
    
    def test_prompt_imagen_contiene_amigable(self, sistema):
        """Debe contener 'amigable' para establecer tono conversacional"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert 'amigable' in prompt.lower(), \
            "El prompt debe mencionar estilo amigable"
    
    def test_prompt_imagen_menciona_imagen_recuperada(self, sistema):
        """Debe mencionar 'imagen recuperada' para contexto de búsqueda"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert 'imagen recuperada' in prompt.lower(), \
            "El prompt debe mencionar imágenes recuperadas"
    
    def test_prompt_imagen_menciona_analisis_visual(self, sistema):
        """Debe mencionar 'análisis visual' para indicar tarea de análisis"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert 'análisis visual' in prompt.lower(), \
            "El prompt debe mencionar análisis visual"
    
    def test_prompt_imagen_contiene_instrucciones_visuales(self, sistema):
        """Debe contener instrucciones específicas para análisis de imágenes"""
        prompt = sistema._generar_prompt_sistema('imagen')
        prompt_lower = prompt.lower()
        # Verificar que contiene términos relacionados con análisis visual
        terminos_visuales = ['imagen', 'visual', 'observa', 'describe']
        encontrados = [t for t in terminos_visuales if t in prompt_lower]
        assert len(encontrados) >= 3, \
            f"El prompt debe contener múltiples términos de análisis visual, encontrados: {encontrados}"
    
    def test_prompt_imagen_contiene_tono_conversacional(self, sistema):
        """Debe mencionar explícitamente 'tono conversacional'"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert 'tono conversacional' in prompt.lower(), \
            "El prompt debe instruir usar tono conversacional"


class TestAmbosPromptsRigorCientifico:
    """Test: ambos prompts mantienen rigor científico (mencionan precisión/evidencia)"""
    
    def test_prompt_texto_menciona_precision(self, sistema):
        """Prompt de texto debe mencionar precisión científica"""
        prompt = sistema._generar_prompt_sistema('texto')
        prompt_lower = prompt.lower()
        # Buscar variantes de precisión
        assert 'precisión' in prompt_lower or 'precis' in prompt_lower, \
            "El prompt de texto debe mencionar precisión"
    
    def test_prompt_imagen_menciona_precision(self, sistema):
        """Prompt de imagen debe mencionar precisión científica"""
        prompt = sistema._generar_prompt_sistema('imagen')
        prompt_lower = prompt.lower()
        # Buscar variantes de precisión
        assert 'precisión' in prompt_lower or 'precis' in prompt_lower, \
            "El prompt de imagen debe mencionar precisión"
    
    def test_prompt_texto_menciona_evidencia_o_contexto(self, sistema):
        """Prompt de texto debe mencionar evidencia o contexto como respaldo"""
        prompt = sistema._generar_prompt_sistema('texto')
        prompt_lower = prompt.lower()
        assert 'evidencia' in prompt_lower or 'contexto' in prompt_lower, \
            "El prompt de texto debe mencionar evidencia o contexto"
    
    def test_prompt_imagen_menciona_evidencia_o_contexto(self, sistema):
        """Prompt de imagen debe mencionar evidencia o contexto como respaldo"""
        prompt = sistema._generar_prompt_sistema('imagen')
        prompt_lower = prompt.lower()
        assert 'evidencia' in prompt_lower or 'contexto' in prompt_lower, \
            "El prompt de imagen debe mencionar evidencia o contexto"
    
    def test_prompt_texto_instruye_no_inventar(self, sistema):
        """Prompt de texto debe instruir no inventar información"""
        prompt = sistema._generar_prompt_sistema('texto')
        prompt_lower = prompt.lower()
        assert 'nunca inventes' in prompt_lower or 'no inventes' in prompt_lower, \
            "El prompt de texto debe instruir no inventar información"
    
    def test_prompt_imagen_instruye_no_inventar(self, sistema):
        """Prompt de imagen debe instruir no inventar información"""
        prompt = sistema._generar_prompt_sistema('imagen')
        prompt_lower = prompt.lower()
        assert 'nunca inventes' in prompt_lower or 'no inventes' in prompt_lower, \
            "El prompt de imagen debe instruir no inventar información"
    
    def test_prompt_texto_mantiene_rigor_cientifico(self, sistema):
        """Prompt de texto debe instruir mantener rigor científico"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert 'rigor científico' in prompt.lower(), \
            "El prompt de texto debe mencionar rigor científico"
    
    def test_prompt_imagen_mantiene_rigor_cientifico(self, sistema):
        """Prompt de imagen debe instruir mantener rigor científico"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert 'rigor científico' in prompt.lower(), \
            "El prompt de imagen debe mencionar rigor científico"


class TestEstructuraPrompts:
    """Tests adicionales para verificar estructura y completitud de prompts"""
    
    def test_prompt_texto_no_vacio(self, sistema):
        """Prompt de texto no debe estar vacío"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert len(prompt.strip()) > 0, "El prompt de texto no debe estar vacío"
    
    def test_prompt_imagen_no_vacio(self, sistema):
        """Prompt de imagen no debe estar vacío"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert len(prompt.strip()) > 0, "El prompt de imagen no debe estar vacío"
    
    def test_prompt_texto_tiene_longitud_razonable(self, sistema):
        """Prompt de texto debe tener longitud suficiente para ser útil"""
        prompt = sistema._generar_prompt_sistema('texto')
        assert len(prompt) > 200, \
            "El prompt de texto debe tener al menos 200 caracteres"
    
    def test_prompt_imagen_tiene_longitud_razonable(self, sistema):
        """Prompt de imagen debe tener longitud suficiente para ser útil"""
        prompt = sistema._generar_prompt_sistema('imagen')
        assert len(prompt) > 200, \
            "El prompt de imagen debe tener al menos 200 caracteres"
    
    def test_prompts_son_diferentes(self, sistema):
        """Los prompts de texto e imagen deben ser diferentes"""
        prompt_texto = sistema._generar_prompt_sistema('texto')
        prompt_imagen = sistema._generar_prompt_sistema('imagen')
        assert prompt_texto != prompt_imagen, \
            "Los prompts de texto e imagen deben ser diferentes"
    
    def test_prompt_texto_contiene_estructura_respuesta(self, sistema):
        """Prompt de texto debe incluir estructura de respuesta esperada"""
        prompt = sistema._generar_prompt_sistema('texto')
        prompt_lower = prompt.lower()
        assert 'estructura de respuesta' in prompt_lower or 'estructura' in prompt_lower, \
            "El prompt de texto debe incluir estructura de respuesta"
    
    def test_prompt_imagen_contiene_estructura_respuesta(self, sistema):
        """Prompt de imagen debe incluir estructura de respuesta esperada"""
        prompt = sistema._generar_prompt_sistema('imagen')
        prompt_lower = prompt.lower()
        assert 'estructura de respuesta' in prompt_lower or 'estructura' in prompt_lower, \
            "El prompt de imagen debe incluir estructura de respuesta"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
