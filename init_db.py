#!/usr/bin/env python3
"""
============================================================================
HISTOLOGÍA RAG - Database Initialization Script
============================================================================
Script dedicado para inicializar la base de datos Qdrant con los PDFs de
histología. Procesa todos los PDFs en ./pdfs/ y crea las colecciones
necesarias usando MUVERA para two-stage retrieval.

Uso: uv run python init_db.py [--clean]

Opciones:
    --clean    Elimina las colecciones existentes antes de crear nuevas
============================================================================
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Importar el asistente desde muvera_test
from muvera_test import (
    AsistenteHistologiaMultimodal,
    limpiar_colecciones,
    QDRANT_URL,
    QDRANT_KEY
)


async def verificar_colecciones(asistente):
    """Verificar y mostrar las colecciones creadas en Qdrant"""
    client = asistente.qdrant_client
    
    collections_esperadas = [
        asistente.gestor_qdrant.content_mv_collection,
        asistente.gestor_qdrant.content_fde_collection
    ]
    
    print("\n" + "="*60)
    print("📊 VERIFICACIÓN DE COLECCIONES")
    print("="*60)
    
    for collection_name in collections_esperadas:
        try:
            info = await client.get_collection(collection_name)
            points_count = info.points_count
            vectors_count = info.vectors_count if hasattr(info, 'vectors_count') else 'N/A'
            print(f"✅ {collection_name}: {points_count} puntos")
        except Exception as e:
            print(f"❌ {collection_name}: No existe o error - {e}")
    
    print("="*60)


async def main():
    """Función principal para inicializar la base de datos"""
    parser = argparse.ArgumentParser(
        description="Inicializar base de datos Qdrant con PDFs de histología"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Eliminar colecciones existentes antes de crear nuevas"
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 HISTOLOGÍA RAG - INICIALIZACIÓN DE BASE DE DATOS")
    print("   🧬 ColBERT + ColPali | 🚀 MUVERA | 📦 Qdrant Cloud")
    print("="*80)
    
    # Verificar credenciales
    if not QDRANT_URL:
        print("❌ Error: QDRANT_URL debe estar configurada en .env")
        sys.exit(1)
    if not QDRANT_URL.startswith('http://localhost') and not QDRANT_KEY:
        print("❌ Error: QDRANT_KEY debe estar configurada para nube en .env")
        sys.exit(1)
    
    print(f"\n📡 Qdrant URL: {QDRANT_URL[:50]}...")
    
    # Buscar PDFs
    pdf_dir = Path("./pdfs")
    if not pdf_dir.exists():
        print(f"❌ Error: No existe el directorio {pdf_dir.absolute()}")
        sys.exit(1)
    
    archivos_pdf = list(pdf_dir.glob("*.pdf"))
    
    if not archivos_pdf:
        print(f"❌ Error: No se encontraron PDFs en {pdf_dir.absolute()}")
        print("   Coloca tus archivos PDF en ./pdfs/ y vuelve a ejecutar")
        sys.exit(1)
    
    print(f"\n📚 PDFs encontrados: {len(archivos_pdf)}")
    for pdf in archivos_pdf:
        print(f"   📄 {pdf.name}")
    
    # Inicializar asistente
    print("\n⏳ Inicializando componentes (esto puede tardar unos minutos)...")
    asistente = AsistenteHistologiaMultimodal()
    asistente.inicializar_componentes()
    
    # Limpiar colecciones si se solicita
    if args.clean:
        print("\n🗑️ Limpiando colecciones anteriores...")
        await limpiar_colecciones(asistente)
        print("✅ Colecciones eliminadas")
    
    # Procesar PDFs
    print("\n🔄 Procesando PDFs con MUVERA (two-stage retrieval)...")
    print("   Esto puede tardar varios minutos dependiendo del tamaño de los PDFs")
    print("   y si hay GPU disponible\n")
    
    await asistente.procesar_y_almacenar_pdfs_multimodal(
        archivos_pdf,
        use_muvera=True
    )
    
    # Verificar colecciones creadas
    await verificar_colecciones(asistente)
    
    print("\n✅ INICIALIZACIÓN COMPLETADA")
    print("   Ahora puedes usar muvera_test.py o la API para hacer consultas")
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
