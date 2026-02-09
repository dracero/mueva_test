"""
Test de Retrieval Accuracy: Toma imágenes al azar de extracted_images_histologia,
las busca en Qdrant usando el pipeline MUVERA (FDE + MV rerank), y valida
que la página correcta aparezca como top result.
"""
import asyncio
import os
import random
import time
import torch
from dotenv import load_dotenv

load_dotenv()

async def test_retrieval():
    from muvera_test import AsistenteHistologiaMultimodal
    
    print("🚀 Inicializando asistente (cargando modelos)...")
    assistant = AsistenteHistologiaMultimodal()
    await assistant.inicializar_componentes()
    
    # List all available page images
    img_dir = "extracted_images_histologia"
    all_images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    all_images.sort()
    
    print(f"\n📂 Imágenes disponibles: {len(all_images)}")
    
    # Pick 5 random images (or all if fewer)
    sample_size = min(5, len(all_images))
    test_images = random.sample(all_images, sample_size)
    
    print(f"🎲 Seleccionadas {sample_size} imágenes al azar:")
    for img in test_images:
        print(f"   - {img}")
    
    # Test each image
    results = []
    
    for img_name in test_images:
        # Extract expected page number from filename (e.g., "arch2.pdf_page_7.png" -> 7)
        expected_page = int(img_name.split("_page_")[1].replace(".png", ""))
        img_path = os.path.join(img_dir, img_name)
        
        print(f"\n{'='*60}")
        print(f"🔍 Buscando: {img_name} (Página esperada: {expected_page})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Use the MUVERA search pipeline
            search_results = await assistant.search_muvera(
                query=None,
                image_path=img_path,
                top_k=5
            )
            
            elapsed = time.time() - start_time
            
            pages = search_results.get("pages", [])
            if not pages:
                print(f"   ❌ No results returned!")
                results.append({"image": img_name, "expected": expected_page, "found": None, "rank": -1, "correct": False})
                continue
            
            # Check where the expected page ranks
            top1_page = pages[0]["payload"].get("page_number")
            top1_score = pages[0].get("score", "N/A")
            
            found_rank = -1
            for rank, r in enumerate(pages, 1):
                if r["payload"].get("page_number") == expected_page:
                    found_rank = rank
                    break
            
            is_correct = (top1_page == expected_page)
            
            print(f"\n   📊 Resultados (Top 5):")
            for rank, r in enumerate(pages, 1):
                pg = r["payload"].get("page_number")
                sc = r.get("score", "N/A")
                marker = " ✅ MATCH" if pg == expected_page else ""
                print(f"      #{rank}: Página {pg} (score: {sc}){marker}")
            
            print(f"\n   ⏱️ Tiempo: {elapsed:.2f}s")
            
            if is_correct:
                print(f"   ✅ CORRECTO — Top-1 es la página esperada ({expected_page})")
            elif found_rank > 0:
                print(f"   ⚠️ PARCIAL — Página esperada ({expected_page}) en posición #{found_rank}, Top-1 fue página {top1_page}")
            else:
                print(f"   ❌ INCORRECTO — Página {expected_page} NO está en Top-5, Top-1 fue página {top1_page}")
            
            results.append({
                "image": img_name,
                "expected": expected_page,
                "top1": top1_page,
                "top1_score": top1_score,
                "rank": found_rank,
                "correct": is_correct,
                "time": f"{elapsed:.2f}s"
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"image": img_name, "expected": expected_page, "found": None, "rank": -1, "correct": False, "error": str(e)})
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE ACCURACY")
    print(f"{'='*60}")
    
    correct = sum(1 for r in results if r.get("correct"))
    in_top5 = sum(1 for r in results if r.get("rank", -1) > 0)
    total = len(results)
    
    print(f"\n   Top-1 Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"   Top-5 Recall:   {in_top5}/{total} ({100*in_top5/total:.0f}%)")
    
    print(f"\n   Detalle:")
    for r in results:
        status = "✅" if r.get("correct") else ("⚠️" if r.get("rank", -1) > 0 else "❌")
        rank_str = f"rank #{r.get('rank')}" if r.get("rank", -1) > 0 else "not in top-5"
        print(f"   {status} {r['image']}: esperada pág {r['expected']}, {rank_str}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
