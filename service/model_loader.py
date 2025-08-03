from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# import os

def load_model(model_path: str, device_preference="auto"):
    is_gpu = torch.cuda.is_available() and device_preference != "cpu"
    print(f"🖥️ Loading model for {'GPU' if is_gpu else 'CPU'}...")

    # Device setup
    device = torch.device("cuda" if is_gpu else "cpu")

    # Optional quantization config for GPU
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ) if is_gpu else None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with or without quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if is_gpu else torch.float32,
        device_map="auto" if is_gpu else None,
        quantization_config=quant_config
    )

    model.to(device)
    model.eval()

    return model, tokenizer
