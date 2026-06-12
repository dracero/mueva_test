import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

# Cargar variables de entorno
load_dotenv()

async def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    qdrant_path = os.getenv("QDRANT_PATH")
    
    print("🧹 Conectando a Qdrant...")
    if qdrant_path:
        client = AsyncQdrantClient(path=qdrant_path, timeout=120)
        print(f"🔗 Cliente Qdrant conectado localmente (ruta: {qdrant_path})")
    else:
        api_key = qdrant_key if qdrant_key else None
        client = AsyncQdrantClient(
            url=qdrant_url or "http://localhost:6333",
            api_key=api_key,
            timeout=120,
            prefer_grpc=False
        )
        print(f"🔗 Cliente Qdrant conectado a {qdrant_url or 'http://localhost:6333'}")

    collection_base = "histopatologia"
    content_mv_collection = f"{collection_base}_content_mv"
    content_fde_collection = f"{collection_base}_content_fde"

    for col in [content_mv_collection, content_fde_collection]:
        try:
            print(f"🗑️ Intentando borrar colección: {col}...")
            await client.delete_collection(col)
            print(f"✅ Colección borrada con éxito: {col}")
        except Exception as e:
            print(f"⚠️ No se pudo borrar o no existía la colección {col}: {e}")
            
    print("\n✨ Proceso de limpieza de base de datos finalizado.")

if __name__ == "__main__":
    asyncio.run(main())
