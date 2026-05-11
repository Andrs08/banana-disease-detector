from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from src.inference.predict import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp", exist_ok=True)

@app.get("/")
def home():
    return {"message": "API funcionando"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    file_path = f"temp/{file.filename}"

    # guardar imagen temporal
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = predict(file_path)
    # eliminar imagen
    os.remove(file_path)

    return result      