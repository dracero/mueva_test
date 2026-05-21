from muvera_test import AsistenteHistologiaMultimodal
import asyncio

async def main():
    print("Iniciando asistente...")
    asistente = AsistenteHistologiaMultimodal()
    print("Asistente inicializado.")

asyncio.run(main())
