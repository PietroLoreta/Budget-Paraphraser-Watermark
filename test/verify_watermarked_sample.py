import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CONFIG_DIR = PROJECT_ROOT/"config"
DATA_DIR = PROJECT_ROOT/"outputs"/"watermarked_dataset"/"paper_conclusion"/"qwen3.5-0.8b"

ALGORITHMS = ["KGW", "SynthID", "Unigram"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=220,
        min_length=230,
        do_sample=True,
        no_repeat_ngram_size=4
    )

    for algorithm_name in ALGORITHMS:
        print(f"\nLoading {algorithm_name}...")

        config_path = CONFIG_DIR / f"{algorithm_name}.json"
        input_path = DATA_DIR / f"{algorithm_name}.jsonl"

        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=str(config_path),
            transformers_config=transformers_config
        )

        with input_path.open("r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        result = watermark.detect_watermark(record["watermarked_text"])

        print(f"arXiv ID: {record["arxiv_id"]}")
        print(f"Detection result: {result}")

if __name__ == "__main__":
    main()