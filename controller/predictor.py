from service.model_loader import load_model
import torch

model, tokenizer = load_model(
    model_path="./modelsAI/tinyllama-lora-extended-v2",
    base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Optional if already merged
)

def generate_prompt(prices, admin_prompt=None):
    price_str = ", ".join(f"{p:.2f}" for p in prices)
    base_prompt = f"Previous prices: {price_str}.\nPredict the next realistic price:"
    if admin_prompt:
        base_prompt += f"\nAdmin Note: {admin_prompt}"
    return base_prompt

def predict_next_price(prices, admin_prompt=None):
    prompt = generate_prompt(prices, admin_prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            max_new_tokens=10,  # ⬅️ reduce this for speed
            repetition_penalty=1.2
        )
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # return output
    predicted_number = extract_price_from_response(output)
    return predicted_number

def extract_price_from_response(response):
    import re
    matches = re.findall(r"\d+\.\d+", response)
    return float(matches[-1]) if matches else None
