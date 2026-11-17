import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.text_quality_analyzer import LLMTextRater

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

PILOT_PATH = PROJECT_ROOT / "outputs" / "dipper_pilot" / "KGW_candidates_36.jsonl"

def main():
    if not PILOT_PATH.exists():
        raise FileNotFoundError(f"File {PILOT_PATH} not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading local quality model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(device)
    model.eval()

    rater = LLMTextRater(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cot=False,
        max_new_tokens=64,
        do_sample=False
    )

    temp_path = PILOT_PATH.with_name(PILOT_PATH.name + ".tmp")
    total = 0

    with PILOT_PATH.open("r", encoding="utf-8") as source, \
            temp_path.open("w", encoding="utf-8") as output:

        for line in source:
            record = json.loads(line)

            source_text = record["source_text"]
            paraphrased_text = record["paraphrased_text"]

            reference = (
                "Evaluate the quality of this paraphrase of an "
                "academic paper conclusion. Consider whether it "
                "preserves the original meaning, key findings and "
                "technical claims, and whether the text is coherent, "
                "grammatically correct and free from unsupported "
                "additions. Do not reward lexical changes alone.\n\n"
                "Original conclusion:\n"
                f"{source_text}"
            )

            score = rater.analyze(
                text=paraphrased_text,
                reference=reference
            )

            record["quality_score"] = float(score)

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            total += 1
            lex_target = record.get(
                "lex_diversity_target",
                record.get("lex_diversity", "n/d")
            )

            print(
                f"Candidate {total}, target lex={lex_target}: "
                f"quality_score={score:.3f}"
            )

    temp_path.replace(PILOT_PATH)

    print(f"{total} candidates rated.")
    print(f"File updated: {PILOT_PATH}")

if __name__ == "__main__":
    main()