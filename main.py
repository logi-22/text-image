from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import pinecone
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, CLIPModel
import numpy as np
import os

# ✅ Initialize FastAPI
app = FastAPI(title="Image & Text Search API", version="1.0")

# ✅ Initialize Pinecone Properly
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key")  # Replace with actual API key
INDEX_NAME = "images-index"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=512,  # CLIP uses 512-dimensional vectors
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-west-2"),
    )

unsplash_index = pc.Index(INDEX_NAME)

# ✅ Load CLIP Model & Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ✅ Function to Generate Embedding from Text
def get_text_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.detach().cpu().numpy().flatten().tolist()


# ✅ Function to Generate Embedding from Image
def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.detach().cpu().numpy().flatten().tolist()


# ✅ Function to Search Pinecone for Similar Images
def search_similar_images(embedding, top_k=10):
    results = unsplash_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="image-search-dataset",
    )
    return results.get("matches", [])


# ✅ API Endpoint: Text-to-Image Search
class TextSearchRequest(BaseModel):
    query: str


@app.post("/search/text")
async def search_by_text(request: TextSearchRequest):
    embedding = get_text_embedding(request.query)
    matches = search_similar_images(embedding, top_k=10)

    if not matches:
        raise HTTPException(status_code=404, detail="No matching images found.")

    return {"query": request.query, "results": matches}


# ✅ API Endpoint: Image-to-Image Search
@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)):
    # Read image file
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    embedding = get_image_embedding(image)
    matches = search_similar_images(embedding, top_k=10)

    if not matches:
        raise HTTPException(status_code=404, detail="No similar images found.")

    return {"filename": file.filename, "results": matches}


# ✅ API Endpoint: Upload Image to Store in Pinecone
@app.post("/store/image")
async def store_image(file: UploadFile = File(...)):
    try:
        # Read image file
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        embedding = get_image_embedding(image)

        # Generate a unique ID (use filename or hash)
        image_id = file.filename

        # Store embedding in Pinecone
        unsplash_index.upsert([(image_id, embedding, {"filename": image_id})])

        return {"message": f"Image {image_id} stored successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# ✅ Health Check Endpoint
@app.get("/")
async def health_check():
    return {"message": "API is running!"}
