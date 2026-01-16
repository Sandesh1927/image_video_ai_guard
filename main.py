from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
import os
import tempfile

# -------------------------
# APP CONFIG (DOCS HIDDEN)
# -------------------------
app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# -------------------------
# SECURITY (API KEY)
# -------------------------
API_KEY = "MY_SUPER_SECRET_KEY_123"

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------------------------
# LOAD MODELS (ONCE)
# -------------------------
yolo_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(["en"], gpu=False)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def extract_links(text: str):
    url_pattern = r"(https?://\S+|www\.\S+)"
    return re.findall(url_pattern, text.lower())

def final_decision(links, text):
    spam_keywords = [
        "win money", "free reward", "click here",
        "urgent", "verify now", "claim now",
        "limited offer", "otp", "bank alert"
    ]

    for word in spam_keywords:
        if word in text.lower():
            return "DANGEROUS"

    if len(links) > 0:
        return "DANGEROUS"

    return "SAFE"

# -------------------------
# IMAGE DETECTION API
# -------------------------
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    # Read image bytes
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # YOLO OBJECT DETECTION
    results = yolo_model(img)
    detected_objects = []

    for r in results:
        for cls in r.boxes.cls:
            detected_objects.append(yolo_model.names[int(cls)])

    # OCR TEXT EXTRACTION
    ocr_results = ocr_reader.readtext(img, detail=0)
    extracted_text = " ".join(ocr_results)

    # LINK DETECTION
    detected_links = extract_links(extracted_text)

    # FINAL DECISION
    decision = final_decision(detected_links, extracted_text)

    return JSONResponse({
        "detected_objects": detected_objects,
        "extracted_text": extracted_text,
        "detected_links": detected_links,
        "final_decision": decision
    })

# -------------------------
# VIDEO DETECTION API
# -------------------------
@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)

    detected_objects = set()
    extracted_text_all = ""

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 15th frame (performance)
        if frame_count % 15 != 0:
            continue

        # YOLO DETECTION
        results = yolo_model(frame)
        for r in results:
            for cls in r.boxes.cls:
                detected_objects.add(yolo_model.names[int(cls)])

        # OCR
        text_results = ocr_reader.readtext(frame, detail=0)
        extracted_text_all += " ".join(text_results) + " "

    cap.release()
    os.remove(video_path)

    # LINK DETECTION
    detected_links = extract_links(extracted_text_all)

    # FINAL DECISION
    decision = final_decision(detected_links, extracted_text_all)

    return JSONResponse({
        "detected_objects": list(detected_objects),
        "extracted_text": extracted_text_all.strip(),
        "detected_links": detected_links,
        "final_decision": decision
    })

# -------------------------
# ROOT CHECK
# -------------------------
@app.get("/")
def root():
    return {"status": "AI Guard Backend Running"}
