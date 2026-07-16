from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("best.pt")

@app.post("/contar-varillas")
async def contar_varillas(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resultado = model.predict(img, conf=0.4, imgsz=960)[0]
    total = len(resultado.boxes)

    return {
        "total_varillas": total,
        "detecciones": [
            {
                "caja": box.tolist(),
                "confianza": float(conf)
            }
            for box, conf in zip(resultado.boxes.xyxy, resultado.boxes.conf)
        ]
    }

@app.get("/")
def health_check():
    return {"status": "servidor activo"}