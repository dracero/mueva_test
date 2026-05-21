from muvera_test import AsistenteHistologiaMultimodal
import asyncio

async def test_indexing():
    asistente = AsistenteHistologiaMultimodal()
    asistente.inicializar_componentes()
    print("Iniciando indexacion...")
    try:
        await asistente.procesar_y_almacenar_pdfs_multimodal(["pdfs/arch3.pdf"], use_muvera=True)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test_indexing())
