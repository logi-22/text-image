from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import pinecone
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, CLIPModel
import numpy as np
import os

# ✅ Initialize FastAPI
app = FastAPI(title="Image & Text Search API", version="1.0")

# ✅ Load CLIP Model & Processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ✅ Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_5W2aRh_QXKhMXdMUC7NVXWPXgpcT6J7cxhpvshc7MYWoPAqpRA8vmwes2Bx2xVnqYXeqme")  # Use env var for security
INDEX_NAME = "images-index"

pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
if INDEX_NAME not in pinecone.list_indexes():
    raise ValueError(f"Pinecone index '{INDEX_NAME}' not found. Please create it first.")

unsplash_index = pinecone.Index(INDEX_NAME)

# ✅ Function to Generate Embedding from Text
def get_text_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy().flatten().tolist()

# ✅ Function to Generate Embedding from Image
def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten().tolist()

# ✅ Function to Search Pinecone for Similar Images
def search_similar_images(embedding, top_k=10):
    try:
        results = unsplash_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.get("matches", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")

# ✅ API Endpoint: Text-to-Image Search
class TextSearchRequest(BaseModel):
    query: str

@app.post("/search/text")
async def search_by_text(request: TextSearchRequest):
    try:
        embedding = get_text_embedding(request.query)
        matches = search_similar_images(embedding, top_k=10)

        if not matches:
            raise HTTPException(status_code=404, detail="No matching images found.")

        return {"query": request.query, "results": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text search: {str(e)}")

# ✅ API Endpoint: Image-to-Image Search
@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        embedding = get_image_embedding(image)
        matches = search_similar_images(embedding, top_k=10)

        if not matches:
            raise HTTPException(status_code=404, detail="No similar images found.")

        return {"filename": file.filename, "results": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image search: {str(e)}")

# ✅ API Endpoint: Upload Image to Store in Pinecone
@app.post("/store/image")
async def store_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        embedding = get_image_embedding(image)

        # Generate a unique ID
        image_id = file.filename

        # Store embedding in Pinecone
        unsplash_index.upsert([(image_id, embedding, {"filename": image_id})])

        return {"message": f"Image {image_id} stored successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing image: {str(e)}")

# ✅ Health Check Endpoint
@app.get("/")
async def health_check():
    return {"message": "API is running!"}
