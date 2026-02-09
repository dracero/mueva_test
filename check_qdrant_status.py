
import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")

async def check_status():
    if not QDRANT_URL:
        print("❌ QDRANT_URL not found in .env")
        return

    print(f"Connecting to {QDRANT_URL}...")
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    
    try:
        collections_response = await client.get_collections()
        collections = collections_response.collections
        
        print(f"\n📊 Qdrant Status: {len(collections)} collections found")
        
        for col in collections:
            name = col.name
            try:
                info = await client.get_collection(name)
                count = info.points_count
                status = info.status
                print(f"   📂 {name:<35} | Count: {count:<6} | Status: {status}")
            except Exception as e:
                print(f"   ⚠️ Could not get info for {name}: {e}")
                
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_status())
