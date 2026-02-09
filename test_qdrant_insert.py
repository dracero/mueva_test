
import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, MultiVectorConfig, MultiVectorComparator

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")

async def test_insert():
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    collection_name = "test_multivector_insert"
    dim = 128
    
    print(f"Connecting to {QDRANT_URL}...")
    
    # Create collection
    try:
        await client.delete_collection(collection_name)
    except:
        pass
        
    print("Creating collection...")
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        )
    )
    
    # Create dummy vector (ColPali style: List[List[float]])
    # 1030 vectors of 128 dims (Real ColPali size)
    import random
    patches = 1030
    dummy_vector = [[0.1] * dim for _ in range(patches)] 
    
    # Simulate a batch of 20 images (approx 10MB)
    batch_size = 20
    points = [
        PointStruct(
            id=i,
            vector=dummy_vector,
            payload={"test": "data"}
        ) for i in range(batch_size)
    ]
    
    # Calc size
    size_mb = (batch_size * patches * dim * 4) / (1024*1024)
    print(f"Inserting {batch_size} points... Vector type: {type(dummy_vector)} (1030x128). Est Size: {size_mb:.2f} MB")
    
    try:
        await client.upsert(collection_name=collection_name, points=points)
        print("✅ Insertion SUCCESS")
    except Exception as e:
        print(f"❌ Insertion FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_insert())
