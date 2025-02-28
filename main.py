from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import os
import logging
from pinecone import Pinecone

# Initialize FastAPI
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set environment variables to avoid permission issues
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache"
os.environ["HF_HOME"] = "/app/.cache"

# Load CLIP model and processor
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    logger.error(f"Error loading CLIP model: {e}")
    raise RuntimeError("Failed to load CLIP model.")

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_6QAd2e_Js1mL941ky9vvGhkGpsGmR7H8aDjKWp2vzpMiRDSvFEFGf5VT6meRJeAft1pNaE"  # Use environment variable
INDEX_NAME = "images-index"

if not PINECONE_API_KEY:
    raise RuntimeError("Pinecone API key not set. Please configure it as an environment variable.")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        raise RuntimeError(f"Pinecone index '{INDEX_NAME}' not found.")
    unsplash_index = pc.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise RuntimeError("Failed to initialize Pinecone.")

# Helper functions for embeddings
def embed_text(text: str):
    try:
        inputs = processor(text=text, return_tensors="pt")
        text_features = model.get_text_features(**inputs)
        return text_features.detach().cpu().numpy().flatten().tolist()
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise HTTPException(status_code=500, detail="Error processing text embedding.")

def embed_image(image: Image.Image):
    try:
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        return image_features.detach().cpu().numpy().flatten().tolist()
    except Exception as e:
        logger.error(f"Error embedding image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image embedding.")

@app.post("/embed_text")
async def search_by_text(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    query_embedding = embed_text(query)

    try:
        search_results = unsplash_index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            namespace="image-search-dataset"
        )
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Error querying Pinecone.")

    if not search_results or not search_results.get("matches"):
        return JSONResponse(content={"message": "No matches found"}, status_code=404)

    return {"results": search_results["matches"]}

@app.post("/embed_image")
async def search_by_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")  # Ensure image is in RGB format
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    query_embedding = embed_image(image)

    try:
        search_results = unsplash_index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            namespace="image-search-dataset"
        )
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Error querying Pinecone.")

    if not search_results or not search_results.get("matches"):
        return JSONResponse(content={"message": "No matches found"}, status_code=404)

    return {"results": search_results["matches"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
