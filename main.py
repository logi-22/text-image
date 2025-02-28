from fastapi import FastAPI, File, UploadFile, HTTPException
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from PIL import Image
import io
from transformers import AutoProcessor, CLIPModel
import numpy as np

app = FastAPI()

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "images-index"
unsplash_index = pc.Index(index_name)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate embedding from text
def get_text_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    embedding = text_features.detach().cpu().numpy().flatten().tolist()
    return embedding

# Function to generate embedding from image
def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    embedding = image_features.detach().cpu().numpy().flatten().tolist()
    return embedding

# Function to query Pinecone and fetch similar images
def search_similar_images(embedding: list, top_k: int = 10):
    results = unsplash_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="image-search-dataset"
    )
    return results["matches"]

@app.get("/search/text/")
async def search_by_text(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    embedding = get_text_embedding(query)
    matches = search_similar_images(embedding)
    return {"matches": [{"id": m["id"], "score": m["score"], "url": m["metadata"]["url"]} for m in matches]}

@app.post("/search/image/")
async def search_by_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        embedding = get_image_embedding(image)
        matches = search_similar_images(embedding)
        return {"matches": [{"id": m["id"], "score": m["score"], "url": m["metadata"]["url"]} for m in matches]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)