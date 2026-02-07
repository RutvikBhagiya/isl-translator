from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import json, os
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque, Counter

app = FastAPI(title="ISL Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = "gestures.json"

def load_db():
    return json.load(open(DATA_FILE)) if os.path.exists(DATA_FILE) else {}

def save_db(db):
    json.dump(db, open(DATA_FILE, "w"))

GESTURES = load_db()

if "" in GESTURES:
    del GESTURES[""]
    save_db(GESTURES)

PRED_WINDOW = deque(maxlen=7)
CONF_THRESHOLD = 0.6


class Landmark(BaseModel):
    x: float
    y: float
    z: float

class PredictRequest(BaseModel):
    landmarks: List[Landmark]

class BatchTrainRequest(BaseModel):
    name: str
    samples: List[List[Landmark]]

def normalize_landmarks(landmarks: List[Landmark]) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in landmarks])
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts /= scale
    return pts.flatten()

@app.post("/predict")
def predict(req: PredictRequest):
    global PRED_WINDOW

    if not GESTURES:
        return {"prediction": None}

    input_vec = normalize_landmarks(req.landmarks)

    best_label = None
    best_score = -1

    for label, samples in GESTURES.items():
        avg_vec = np.mean(np.array(samples), axis=0)

        score = cosine_similarity(
            input_vec.reshape(1, -1),
            avg_vec.reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_label = label

    if best_score < CONF_THRESHOLD:
        return {"prediction": None}

    PRED_WINDOW.append(best_label)
    final_prediction = Counter(PRED_WINDOW).most_common(1)[0][0]

    return {
        "prediction": final_prediction,
        "confidence": round(best_score * 100, 2),
        "window": list(PRED_WINDOW)
    }


@app.post("/train-batch")
def train_batch(req: BatchTrainRequest):
    name = req.name.strip().upper()
    if not name:
        return {"error": "INVALID_NAME"}

    if name not in GESTURES:
        GESTURES[name] = []

    for s in req.samples:
        GESTURES[name].append(normalize_landmarks(s).tolist())

    save_db(GESTURES)
    return {"status": "ok", "total": len(GESTURES[name])}

@app.get("/gestures")
def gesture_count():
    return {
        "count": len(GESTURES),
        "names": list(GESTURES.keys())
    }
