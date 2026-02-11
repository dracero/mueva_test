
import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.getenv("QDRANT_KEY", None)

async def clean():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    
    collections = ["histopatologia_content_mv", "histopatologia_content_fde"]
    
    for col in collections:
        try:
            print(f"Deleting collection: {col}...")
            await client.delete_collection(col)
            print(f"✅ Deleted {col}")
        except Exception as e:
            print(f"⚠️ Error deleting {col}: {e}")

if __name__ == "__main__":
    asyncio.run(clean())
