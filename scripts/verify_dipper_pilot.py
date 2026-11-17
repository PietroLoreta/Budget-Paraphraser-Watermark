import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CONFIG_DIR = PROJECT_ROOT / "config"
PILOT_PATH = PROJECT_ROOT / "outputs" / "dipper_pilot" / "KGW_candidates_36.jsonl"

def main():
    if not PILOT_PATH.exists():
        raise FileNotFoundError(f"File not found: {PILOT_PATH}")

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

    print("Loading KGW detector...")
    watermark = AutoWatermark.load(
        "KGW",
        algorithm_config=str(CONFIG_DIR / "KGW.json"),
        transformers_config=transformers_config
    )

    temp_path = PILOT_PATH.with_name(PILOT_PATH.name + ".tmp")

    total = 0
    detected = 0

    with PILOT_PATH.open("r", encoding="utf-8") as source, \
        temp_path.open("w", encoding="utf-8") as output:

        for line in source:
            record = json.loads(line)

            result = watermark.detect_watermark(record["paraphrased_text"])

            record["watermark_detected"] = bool(result["is_watermarked"])
            record["watermark_detection_score"] = float(result["score"])

            total += 1
            if record["watermark_detected"]:
                detected += 1

            output.write(json.dumps(record, ensure_ascii=False) + "\n")

            lex_target = record.get(
                "lex_diversity_target",
                record.get("lex_diversity", "n/d")
            )

            print(
                f"Target lex={lex_target}: "
                f"detected={record['watermark_detected']}, "
                f"score={record['watermark_detection_score']:.4f}"
            )

    temp_path.replace(PILOT_PATH)

    rate = 100 * detected / total if total else 0.0
    print(f"\nKGW detected: {detected}/{total} ({rate:.2f}%)")
    print(f"Updated file: {PILOT_PATH}")

if __name__ == "__main__":
    main()