import os
import io
import time
import base64
import logging
import sys
import numpy as np
import cv2
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import insightface
from insightface.app import FaceAnalysis

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("InsightFaceService")

app = FastAPI(
    title="InsightFace ArcFace 512D Service",
    description="High-accuracy face detection and 512D ArcFace embedding extraction service.",
    version="1.0.0"
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"-> {request.method} {request.url.path} from {client_ip}")
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(f"<- {request.method} {request.url.path} status={response.status_code} completed in {duration:.2f}ms")
    return response

# Enable CORS for frontend and PHP integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FaceAnalysis instance
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        logger.info("Initializing models (buffalo_s / CPU)...")
        try:
            face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Models (buffalo_s) loaded and initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalysis models: {e}", exc_info=True)
            raise
    return face_app

def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def decode_base64_image(b64_string: str) -> np.ndarray:
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return decode_image_bytes(image_bytes)

class Base64ExtractRequest(BaseModel):
    image: str # Base64 encoded string

class VerifyRequest(BaseModel):
    embedding1: List[float]
    embedding2: List[float]
    threshold: Optional[float] = 0.50

@app.on_event("startup")
def startup_event():
    try:
        get_face_app()
    except Exception as e:
        logger.warning(f"Warning during startup model load: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "service": "InsightFace ArcFace 512D Service", "model": "buffalo_s"}

@app.post("/extract")
async def extract_embedding(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Extracts 512-dimensional ArcFace embedding from uploaded image file or base64 string.
    """
    img = None
    if file is not None:
        contents = await file.read()
        img = decode_image_bytes(contents)
        logger.info(f"Processing uploaded image file: {file.filename} ({len(contents)} bytes)")
    elif image_base64 is not None and image_base64.strip():
        img = decode_base64_image(image_base64)
        logger.info(f"Processing base64 image data ({len(image_base64)} chars)")
    else:
        logger.warning("Extraction failed: No image file or base64 data provided.")
        raise HTTPException(status_code=400, detail="No image file or base64 provided.")

    if img is None:
        logger.warning("Extraction failed: Could not decode image data.")
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    try:
        engine = get_face_app()
        faces = engine.get(img)
    except Exception as e:
        logger.error(f"Inference error during face detection/embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if len(faces) == 0:
        logger.warning("Detection result: No face detected in the provided image.")
        return {
            "success": False,
            "detail": "No face detected. Please ensure your face is clearly visible."
        }

    if len(faces) > 1:
        # Sort by bounding box area (largest face first)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        # If secondary face is significant, warn user
        main_area = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
        second_area = (faces[1].bbox[2] - faces[1].bbox[0]) * (faces[1].bbox[3] - faces[1].bbox[1])
        if second_area > 0.4 * main_area:
            logger.warning(f"Detection result: Multiple faces detected ({len(faces)} faces).")
            return {
                "success": False,
                "detail": "Multiple faces detected. Please show only one face in frame."
            }

    face = faces[0]
    raw_emb = face.embedding
    # L2 normalize the embedding
    norm = np.linalg.norm(raw_emb)
    if norm > 0:
        normalized_emb = (raw_emb / norm).tolist()
    else:
        normalized_emb = raw_emb.tolist()

    det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
    gender = int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None
    age = int(face.age) if hasattr(face, 'age') and face.age is not None else None

    logger.info(f"Extraction successful: det_score={det_score:.3f}, age={age}, gender={gender}, embedding_dim={len(normalized_emb)}")

    return {
        "success": True,
        "embedding": normalized_emb, # 512-float vector
        "embedding_size": len(normalized_emb),
        "det_score": det_score,
        "bbox": [float(x) for x in face.bbox],
        "gender": gender,
        "age": age
    }

@app.post("/extract-base64")
async def extract_embedding_base64_json(payload: Base64ExtractRequest):
    img = decode_base64_image(payload.image)
    if img is None:
        logger.warning("extract-base64 failed: Invalid base64 image.")
        raise HTTPException(status_code=400, detail="Failed to decode image from base64.")

    engine = get_face_app()
    faces = engine.get(img)

    if len(faces) == 0:
        logger.warning("extract-base64 result: No face detected.")
        return {"success": False, "detail": "No face detected."}

    face = faces[0]
    raw_emb = face.embedding
    norm = np.linalg.norm(raw_emb)
    normalized_emb = (raw_emb / norm).tolist() if norm > 0 else raw_emb.tolist()
    det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0

    logger.info(f"extract-base64 successful: det_score={det_score:.3f}, embedding_dim={len(normalized_emb)}")

    return {
        "success": True,
        "embedding": normalized_emb,
        "embedding_size": len(normalized_emb),
        "det_score": det_score,
        "bbox": [float(x) for x in face.bbox]
    }

@app.post("/verify")
def verify_embeddings(payload: VerifyRequest):
    """
    Computes cosine similarity between two 512D embeddings.
    """
    vec1 = np.array(payload.embedding1, dtype=np.float32)
    vec2 = np.array(payload.embedding2, dtype=np.float32)

    if vec1.shape != vec2.shape:
        logger.warning(f"Verify failed: Dimension mismatch ({vec1.shape} vs {vec2.shape})")
        raise HTTPException(status_code=400, detail="Embedding dimensions do not match.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))

    is_match = similarity >= payload.threshold
    logger.info(f"Verify result: similarity={similarity:.4f}, threshold={payload.threshold}, is_match={is_match}")

    return {
        "is_match": is_match,
        "similarity": similarity,
        "threshold": payload.threshold,
        "confidence": f"{max(0.0, similarity) * 100:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://127.0.0.1:8001 ...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
