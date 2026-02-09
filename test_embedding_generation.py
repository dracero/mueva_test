
import asyncio
import os
import numpy as np
from PIL import Image
from muvera_test import AsistenteHistologiaMultimodal

# Mock environment if needed (though muvera_test loads .env)
# Ensure we run in the same dir

async def test_generation():
    print("🚀 Initializing Assistant (loading models)...")
    assistant = AsistenteHistologiaMultimodal()
    
    # Initialize only embedding models to save time/resources if possible, 
    # but the class initializes them in `inicializar_componentes` or lazily?
    # Actually `inicializar_componentes` calls `_inicializar_modelos_embedding`
    
    # We need to manually trigger initialization if it's not in __init__
    # The class calls it? No, in `api.py` startup_event calls it.
    
    assistant._inicializar_modelos_embedding()
    
    print("🖼️ Creating dummy image...")
    img = Image.new('RGB', (224, 224), color = 'red')
    img_path = "test_dummy_image.png"
    img.save(img_path)
    
    print(f"🧠 Generating embedding for {img_path}...")
    try:
        embedding = assistant.generate_image_embedding(img_path)
        
        if embedding is None:
            print("❌ Embedding is None!")
        else:
            print(f"✅ Embedding generated.")
            print(f"   Type: {type(embedding)}")
            if isinstance(embedding, list):
                print(f"   Length (outer): {len(embedding)}")
                if len(embedding) > 0:
                    print(f"   Inner Type: {type(embedding[0])}")
                    if isinstance(embedding[0], list):
                        print(f"   Inner Length: {len(embedding[0])}")
                        print(f"   First 5 values of first vector: {embedding[0][:5]}")
                        
                        # Check for NaNs
                        arr = np.array(embedding)
                        if np.isnan(arr).any():
                            print("   ⚠️ WARNING: NaN values detected!")
                        if np.isinf(arr).any():
                            print("   ⚠️ WARNING: Inf values detected!")
                            
            # Clean up
            if os.path.exists(img_path):
                os.remove(img_path)
                
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generation())
