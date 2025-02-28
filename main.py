from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from PIL import Image
import io
from transformers import AutoProcessor, CLIPModel
import numpy as np
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# JWT Config
SECRET_KEY = os.getenv("JWT_SECRET", "default_secret")  # Use a secure secret in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Fake user database (replace with real authentication logic)
fake_users_db = {
    "admin": {
        "username": "admin",
        "password": "password123"  # Replace with hashed password in production
    }
}

# Initialize FastAPI
app = FastAPI()

# Load Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Please set it in the environment or .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "images-index"
unsplash_index = pc.Index(index_name)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  # Ensure model is in evaluation mode

# OAuth2 authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or user["password"] != password:
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")


@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


def get_text_embedding(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    return text_features.detach().cpu().numpy().flatten().tolist()


def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features.detach().cpu().numpy().flatten().tolist()


def search_similar_images(embedding: list, top_k: int = 10):
    results = unsplash_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="image-search-dataset"
    )
    return results["matches"]


@app.get("/search/text/")
async def search_by_text(query: str, user: str = Depends(get_current_user)):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    embedding = get_text_embedding(query)
    matches = search_similar_images(embedding)
    return {"matches": [{"id": m["id"], "score": m["score"], "url": m["metadata"]["url"]} for m in matches]}


@app.post("/search/image/")
async def search_by_image(file: UploadFile = File(...), user: str = Depends(get_current_user)):
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
