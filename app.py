from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.models import load_model
import cv2
import numpy as np

app = FastAPI(title="Face Emotion Detection API")

# Load model and face detector once
model = load_model("face_emotion.h5")
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}


@app.get("/")
def home():
    return {"message": "Face Emotion Detection API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 3)

        if len(faces) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No face detected"}
            )

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        prediction = model.predict(face, verbose=0)

        emotion_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "emotion": labels[emotion_id],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )