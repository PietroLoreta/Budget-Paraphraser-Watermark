import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from transformers import AutoConfig, T5Tokenizer, T5ForConditionalGeneration
from evaluation.tools.text_editor import DipperParaphraser
from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity

INPUT_PATH = PROJECT_ROOT /"outputs"/"watermarked_dataset"/"paper_conclusion"/"qwen3.5-0.8b"/"KGW.jsonl"
TOKENIZER_PATH = "google/t5-v1_1-xxl"
DIPPER_MODEL_PATH = "/hdd10t/n333884/models/dipper-paraphraser-xxl"
OUTPUT_PATH = PROJECT_ROOT/"outputs"/"dipper_pilot"/"KGW_candidates.jsonl"

MAX_SOURCE_RECORDS = 73
DIVERSITY_CONFIGS = [(20,0), (40,0), (60,0), (40,20), (60,20), (60,40)]
REPEATS_PER_TARGET = 5

def main():
    completed_ids = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)

                if "candidate_id" not in record:
                    raise RuntimeError(
                        f"Record {line_number} without candidate_id: "
                        "cannot proceed safely."
                    )

                completed_ids.add(record["candidate_id"])

        print(f"Already saved records: {len(completed_ids)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    source_records = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                source_records.append(json.loads(line))
            if len(source_records) == MAX_SOURCE_RECORDS:
                break

    if not source_records:
        raise RuntimeError("No source records found")

    print(f"Original records: {len(source_records)}")

    print("Loading DIPPER tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)

    print("Loading DIPPER model config...")
    model_config = AutoConfig.from_pretrained(DIPPER_MODEL_PATH)
    model_config.tie_word_embeddings = False

    print("Loading DIPPER model...")
    model = T5ForConditionalGeneration.from_pretrained(
        DIPPER_MODEL_PATH,
        config=model_config,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": "100GiB"}
    )
    model.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for source_idx, source_record in enumerate(source_records, start = 1):
            source_text = source_record["watermarked_text"]
            arxiv_id = source_record["arxiv_id"]

            for lex_target, order_target in DIVERSITY_CONFIGS:
                print(
                    f"\nTesto {source_idx}/{len(source_records)}"
                    f" | lex={lex_target} "
                    f" | order={order_target}"
                )

                paraphraser = DipperParaphraser(
                    tokenizer=tokenizer,
                    model=model,
                    device="cuda",
                    lex_diversity=lex_target,
                    order_diversity=order_target,
                    sent_interval=3,
                    max_length=512,
                    do_sample=True,
                    top_p=0.75,
                    top_k=0
                    )

                for repeat in range(1, REPEATS_PER_TARGET + 1):
                    candidate_id = f"{arxiv_id}_lex{lex_target}_ord{order_target}_rep{repeat}"

                    if candidate_id in completed_ids:
                        print(f"Already completed, skip: {candidate_id}")
                        continue

                    print(f"Generating {repeat}/{REPEATS_PER_TARGET}")

                    paraphrased_text = paraphraser.edit(
                            source_text,
                            reference=""
                        )

                    if not paraphrased_text or not paraphrased_text.strip():
                        print("WARNING: paraphrased text is empty")
                        continue

                    paraphrased_text = paraphrased_text.strip()

                    result = {
                        "candidate_id": candidate_id,
                        "arxiv_id": arxiv_id,
                        "source_algorithm": "KGW",
                        "repeat": repeat,

                        "lex_diversity_target": lex_target,
                        "order_diversity_target": order_target,

                        "lex_diversity_actual": lexical_diversity(source_text, paraphrased_text),
                        "order_diversity_actual": order_diversity(source_text, paraphrased_text),

                        "quality_score": None,
                        "watermark_detected": None,

                        "source_text": source_text,
                        "paraphrased_text": paraphrased_text
                    }

                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()

                    completed_ids.add(candidate_id)
                    total += 1

                    print(
                        f"Actual lex={result['lex_diversity_actual']}, "
                        f"order={result['order_diversity_actual']}"
                    )

    print(f"\nNew candidates generated: {total}")
    print(f"Total candidates saved: {len(completed_ids)}")
    print(f"File: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()