# services/load_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(model_path: str, base_model_path: str):
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    print("🔌 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)

    print("🧠 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Optional: Compile model for speed (PyTorch 2.0+)
    try:
        print("🚀 Compiling model for faster CPU inference...")
        model = torch.compile(model)
    except Exception as e:
        print("⚠️ torch.compile failed or not available:", str(e))

    model.eval()

    return model, tokenizer

