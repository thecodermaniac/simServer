from service.model_loader import load_model
import torch
import re

# Set to "cpu" or "gpu" or "auto"
DEVICE = "cpu"  # Change to "gpu" / "cpu" based on pytorch setup.
MODEL_PATH = "./modelsAI/tinyllama-merged-v3-cpu" if DEVICE == "cpu" else "./modelsAI/tinyllama-merged-v3-gpu"

model, tokenizer = load_model(MODEL_PATH, device_preference=DEVICE)

def generate_prompt(prices, admin_prompt):
    price_str = ", ".join(f"{p:.2f}" for p in prices)
    base_prompt = f"Previous prices: {price_str}.\nMarket Sentiment: {admin_prompt}\nPredict the next realistic price:"
    return base_prompt

def extract_prices_from_response(response: str):
    # Find the specific line that starts with our expected prefix
    match = re.search(r"Predict the next realistic price:\s*(.*)", response, re.IGNORECASE)
    if not match:
        return []

    price_line = match.group(1)

    # Extract float numbers from the matched line only
    prices = re.findall(r"\d+\.\d+", price_line)
    return [float(p) for p in prices]

def predict_next_price(prices, admin_prompt, max_tokens=40):
    prompt = generate_prompt(prices, admin_prompt)
    # print(f"Generated prompt: {prompt}") # Uncomment for debugging
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"Generated response: {output}") # Uncomment for debugging
    return extract_prices_from_response(output)
