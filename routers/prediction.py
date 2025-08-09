from fastapi import APIRouter, HTTPException
from models.schemas import PriceRequest, PriceResponse
from controller.predictor import predict_next_price


router = APIRouter()

@router.post("/predict", response_model=PriceResponse)
def predict_price(data: PriceRequest):
    if len(data.history) < 5:
        raise HTTPException(status_code=400, detail="At least 5 price points required.")
    if not data.prompt:
        raise HTTPException(status_code=400, detail="prompt is required.")
    prediction = predict_next_price(data.history, data.prompt)
    
    if prediction is None:
        raise HTTPException(status_code=500, detail="Model could not generate a valid number.")
    
    print(f"Received history: {data.history}")
    print(f"Received prompt: {data.prompt}")
    print(f"Prediction: {prediction}")
    return PriceResponse(predicted_price=prediction)