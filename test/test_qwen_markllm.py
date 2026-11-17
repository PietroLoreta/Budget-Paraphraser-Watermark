import sys
from pathlib import Path

#Identifica la cartella principale del progetto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B").to(device)
#Imposta il modello in modalità di valutazione
model.eval()

transformers_config = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=200,
    do_sample=True,
    no_repeat_ngram_size=4
)

print("Loading KGW...")
myWatermark = AutoWatermark.load(
    "KGW",
    algorithm_config="config/KGW.json",
    transformers_config=transformers_config
)

prompt = "Explain why reproducible experiments are important in scientific research."

print("\nGenerating watermarked text...")
watermarked_text = myWatermark.generate_watermarked_text(prompt)

print("\n--- WATERMARKED TEXT ---")
print(watermarked_text)

print("\nVerifying watermarked text...")
result = myWatermark.detect_watermark(watermarked_text)
print(result)

print("\nGenerating unwatermarked text...")
unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

print("\n--- UNWATERMARKED TEXT ---")
print(unwatermarked_text)

print("\nVerifying unwatermarked text...")
control_result = myWatermark.detect_watermark(unwatermarked_text)
print(control_result)