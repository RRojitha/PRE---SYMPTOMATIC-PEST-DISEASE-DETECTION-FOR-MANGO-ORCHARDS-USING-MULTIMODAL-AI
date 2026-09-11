from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from predictor import predict
from predictor_fusion import predict_fusion

app = FastAPI(title="Mango Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    image = Image.open(file.file)

    result = predict(image)

    return result


@app.post("/predict-fusion")
async def predict_image_fusion(
    file: UploadFile = File(...),
    temperature_c: float = Form(...),
    humidity_percent: float = Form(...),
    wind_speed_kmh: float = Form(...),
):

    if not (-10 <= temperature_c <= 55):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter: temperature_c must be between -10 and 55 (got {temperature_c})",
        )
    if not (0 <= humidity_percent <= 100):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter: humidity_percent must be between 0 and 100 (got {humidity_percent})",
        )
    if not (0 <= wind_speed_kmh <= 150):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter: wind_speed_kmh must be between 0 and 150 (got {wind_speed_kmh})",
        )

    image = Image.open(file.file)

    result = predict_fusion(image, temperature_c, humidity_percent, wind_speed_kmh)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)