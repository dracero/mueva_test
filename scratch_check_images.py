import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10)
    col_name = "histopatologia_content_mv"
    
    # Let's see what pages have images
    res = await client.scroll(
        collection_name=col_name,
        scroll_filter=Filter(must=[FieldCondition(key="tipo", match=MatchValue(value="imagen"))]),
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    images = res[0]
    print(f"Found {len(images)} images in DB (limit 100).")
    pages_with_images = set()
    for img in images:
        p = img.payload
        pages_with_images.add(p.get('numero_pagina'))
    print(f"Pages with images: {sorted(list(pages_with_images))}")

if __name__ == "__main__":
    asyncio.run(main())
