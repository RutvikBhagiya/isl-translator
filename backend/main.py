from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="ISL Instant Translator API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SIMILARITY_THRESHOLD = 0.92
EXPECTED_LANDMARKS = 21 
GESTURES: Dict[str, List[np.ndarray]] = {}


class Landmark(BaseModel):
    x: float
    y: float
    z: float

class LandmarkRequest(BaseModel):
    landmarks: List[Landmark] = Field(..., min_items=EXPECTED_LANDMARKS)

class TrainRequest(LandmarkRequest):
    name: str = Field(..., min_length=1, max_length=50)

def normalize_landmarks(landmarks: List[Landmark]) -> np.ndarray:
    """
    Converts landmarks into a normalized vector.
    Uses wrist as origin to remove translation variance.
    """
    base_x = landmarks[0].x
    base_y = landmarks[0].y

    vector = []
    for point in landmarks:
        vector.extend([
            point.x - base_x,
            point.y - base_y,
            point.z
        ])

    return np.array(vector, dtype=np.float32)

def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/train")
def train_gesture(req: TrainRequest):
    vector = normalize_landmarks(req.landmarks)

    if req.name not in GESTURES:
        GESTURES[req.name] = []

    GESTURES[req.name].append(vector)

    print(f"[TRAIN] {req.name} → samples: {len(GESTURES[req.name])}")

    return {
        "status": "saved",
        "gesture": req.name,
        "samples": len(GESTURES[req.name])
    }

@app.post("/predict")
def predict_gesture(req: LandmarkRequest):

    if not GESTURES:
        return {"gesture": "NO_GESTURE"}

    input_vector = normalize_landmarks(req.landmarks)

    best_match = "UNKNOWN"
    best_score = -1.0

    for gesture_name, samples in GESTURES.items():
        avg_vector = np.mean(samples, axis=0)
        score = cosine_score(input_vector, avg_vector)

        if score > best_score:
            best_score = score
            best_match = gesture_name

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "gesture": "UNKNOWN",
            "confidence": round(best_score * 100, 2)
        }

    return {
        "gesture": best_match,
        "confidence": round(best_score * 100, 2)
    }
