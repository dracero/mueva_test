"""
calibrate_threshold.py — Calibrador de umbral para ColPali MaxSim

USO:
    # Con imagen CORRECTA (que SÍ está en la BD):
    python calibrate_threshold.py uploads/imagen_correcta.jpg

    # Con imagen INCORRECTA (que NO está en la BD):
    python calibrate_threshold.py uploads/imagen_falsa.jpg

El script muestra los top-20 scores para que veas la distribución
y puedas elegir un threshold que separe correcto de incorrecto.

OBJETIVO:
    score(correcta) > THRESHOLD > score(incorrecta)

En GTX 1070 (sin normalización L2): correcto=878, incorrecto=846 → threshold=868
En RTX 3050 (RTX, con/sin L2): mídelo con este script.
"""

import asyncio
import os
import sys
import torch

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA: {torch.version.cuda}")
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cudnn:  {torch.backends.cudnn.allow_tf32}")
print()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muvera_test import AsistenteHistologiaMultimodal, Config

async def main():
    if len(sys.argv) < 2:
        print("USO: python calibrate_threshold.py <ruta_imagen>")
        print()
        print("Configuración actual:")
        print(f"  SEARCH_SCORE_THRESHOLD = {Config.SEARCH_SCORE_THRESHOLD}")
        print(f"  NORMALIZE_EMBEDDINGS   = {Config.NORMALIZE_EMBEDDINGS}")
        print(f"  TOP_K_RESULTS          = {Config.TOP_K_RESULTS}")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Imagen no encontrada: {img_path}")
        sys.exit(1)

    print(f"📸 Imagen de prueba: {img_path}")
    print(f"🔧 NORMALIZE_EMBEDDINGS = {Config.NORMALIZE_EMBEDDINGS}")
    print(f"🔧 SEARCH_SCORE_THRESHOLD = {Config.SEARCH_SCORE_THRESHOLD}")
    print()

    asistente = AsistenteHistologiaMultimodal()
    asistente.inicializar_componentes()

    print("\n📊 Generando embedding de la imagen de query...")
    query_mv = asistente.procesador.generar_embedding_imagen(img_path)
    if query_mv is None:
        print("❌ No se pudo generar embedding")
        return

    query_fde = asistente.procesador.generar_query_muvera(query_mv)

    print("\n🔍 Buscando en Qdrant (top-20 sin umbral)...")
    # Busca con min_score=0 para ver TODOS los candidatos
    resultados, _ = await asistente.gestor_qdrant.buscar_muvera_2stage(
        query_mv, query_fde, top_k=20, min_score=0.0
    )

    if not resultados:
        print("⚠️ No se encontraron resultados. ¿Está la base de datos indexada?")
        return

    print(f"\n{'='*65}")
    print(f"{'#':<4} {'Score':>10} {'Tipo':<8} {'Pág':>5}  Archivo")
    print(f"{'='*65}")
    for i, r in enumerate(resultados):
        p = r.get('payload', {})
        score = r['score']
        tipo = p.get('tipo', '?')
        pag = p.get('numero_pagina', '?')
        nombre = p.get('nombre_archivo', p.get('pdf_name', '?'))
        marker = " ◀ TOP-1" if i == 0 else ""
        print(f"{i+1:<4} {score:>10.4f}  {tipo:<8} {str(pag):>5}  {nombre}{marker}")

    scores = [r['score'] for r in resultados]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else top1
    gap = top1 - top2

    print(f"\n{'='*65}")
    print(f"📈 ANÁLISIS:")
    print(f"   Top-1 score   : {top1:.4f}")
    print(f"   Top-2 score   : {top2:.4f}")
    print(f"   Gap 1→2       : {gap:.4f}  ({gap/top1*100:.1f}% del top-1)")
    print(f"   Punto medio   : {(top1+top2)/2:.4f}  ← candidato a threshold")
    print()
    print(f"💡 PARA CALIBRAR:")
    print(f"   Si esta imagen ES correcta (está en BD):")
    print(f"     → Anota su score: {top1:.4f}")
    print(f"     → Repeat con imagen INCORRECTA y anota su top-1 score")
    print(f"     → Pon el threshold entre los dos en .env:")
    print(f"        SEARCH_SCORE_THRESHOLD=<valor_entre_ambos>")
    print()
    print(f"   Threshold actual en .env: {Config.SEARCH_SCORE_THRESHOLD}")

if __name__ == "__main__":
    asyncio.run(main())
