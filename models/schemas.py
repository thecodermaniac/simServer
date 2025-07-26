from pydantic import BaseModel
from typing import List, Optional

class PriceRequest(BaseModel):
    history: List[float]  # last 10 prices
    prompt: Optional[str] = None  # optional admin prompt

class PriceResponse(BaseModel):
    predicted_price: float
