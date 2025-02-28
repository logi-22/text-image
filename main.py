from fastapi import FastAPI, HTTPException, Depends, UploadFile, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
import io
import numpy as np
from transformers import AutoProcessor, CLIPModel
from pinecone import Pinecone
import logging

app = FastAPI()
security = HTTPBasic()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dummy credentials (Replace with a proper authentication system)
CREDENTIALS = {"admin": "password123"}

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_6QAd2e_Js1mL941ky9vvGhkGpsGmR7H8aDjKWp2vzpMiRDSvFEFGf5VT6meRJeAft1pNaE")
index_name = "images-index"

# Ensure Pinecone index exists
index_list = pc.list_indexes().names()
if index_name not in index_list:
    raise RuntimeError(f"Index '{index_name}' not found. Make sure it is created.")

# Initialize Pinecone index
unsplash_index = pc.Index(index_name)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username in CREDENTIALS and CREDENTIALS[credentials.username] == credentials.password:
        return credentials.username
    raise HTTPException(status_code=401, detail="Invalid username or password")

def embed_text(text: str):
    inputs = processor(text=text, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features.detach().cpu().numpy().flatten().tolist()

def embed_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features.detach().cpu().numpy().flatten().tolist()

@app.post("/embed_text/")
def embed_text_api(text: str = Form(...), username: str = Depends(authenticate)):
    embedding = embed_text(text)
    return {"embedding": embedding}

@app.post("/embed_image/")
def embed_image_api(file: UploadFile, username: str = Depends(authenticate)):
    image = Image.open(io.BytesIO(file.file.read()))
    embedding = embed_image(image)
    return {"embedding": embedding}

@app.post("/search/")
def search(query_embedding: list, username: str = Depends(authenticate)):
    search_results = unsplash_index.query(
        vector=query_embedding, top_k=10, include_metadata=True, namespace="image-search-dataset"
    )
    return {"results": search_results.get("matches", [])}
