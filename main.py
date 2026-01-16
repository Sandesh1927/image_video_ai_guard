from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
from tempfile import NamedTemporaryFile
import os

# =========================
# APP CONFIG (NO /docs)
# =========================
app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# =========================
# LOAD MODELS (ONCE)
# =========================
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)

# =========================
# RULES
# =========================
BLOCKED_OBJECTS = ["knife", "gun", "fire", "weapon"]
SUSPICIOUS_WORDS = ["win", "free", "click", "verify", "urgent"]

# =========================
# IMAGE DETECTION API
# =========================
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)
    detected_objects = []
    status = "SAFE"

    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        detected_objects.append(label)

        if label in BLOCKED_OBJECTS:
            status = "BLOCKED"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texts = reader.readtext(gray, detail=0)
    full_text = " ".join(texts).lower()

    links = re.findall(r'(https?://\S+|www\.\S+)', full_text)

    for word in SUSPICIOUS_WORDS:
        if word in full_text:
            status = "SUSPICIOUS"

    if links:
        status = "BLOCKED"

    return {
        "objects": detected_objects,
        "text": full_text,
        "links": links,
        "final_status": status
    }

# =========================
# VIDEO DETECTION API  ✅ ADD THIS BELOW IMAGE API
# =========================
@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(await file.read())
        video_path = temp.name

    cap = cv2.VideoCapture(video_path)

    blocked = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]

            if label in BLOCKED_OBJECTS:
                blocked = True
                break

        if blocked:
            break

    cap.release()
    os.remove(video_path)

    status = "BLOCKED" if blocked else "SAFE"

    return {
        "final_status": status
    }
