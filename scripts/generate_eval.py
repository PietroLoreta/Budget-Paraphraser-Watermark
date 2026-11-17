import gc
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity
from evaluation.tools.text_quality_analyzer import LLMTextRater
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

INPUT_PATH = PROJECT_ROOT/"outputs"/"watermarked_dataset"/"paper_conclusion"/"qwen3.5-0.8b"/"KGW.jsonl"
OUTPUT_PATH = PROJECT_ROOT/"outputs"/"step4"/"KGW_test_generations.jsonl"
KGW_CONFIG_PATH = PROJECT_ROOT/"config"/"KGW.json"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "/hdd10t/n333884/models/qwen2.5-3b-dpo-diversity"
SCORER_MODEL_NAME = "Qwen/Qwen3.5-0.8B"

TRAIN_SIZE = 50                                   # i primi 50 testi sono del training
TARGETS = [(20, 0), (40, 0), (60, 0), (40, 20), (60, 20), (60, 40)]
REPEATS = 2
MAX_NEW_TOKENS = 450

SYSTEM_PROMPT = (
    "You are an expert copy-editor. Paraphrase the text you are given, preserving its "
    "meaning. Do not respond with a greeting or any extraneous information."
)

def build_user_prompt(text, lex, order):
    return (
        "Paraphrase the following academic conclusion while preserving its "
        "scientific meaning, findings, and technical claims. Follow the "
        "requested diversity targets as closely as possible.\n"
        f"Target lexical diversity: {lex}/100.\n"
        f"Target order diversity: {order}/100.\n\n"
        "[[START OF TEXT]]\n"
        f"{text}\n"
        "[[END OF TEXT]]\n"
        "[[START OF PARAPHRASE]]\n"
    )

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl_atomic(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_all(test_records):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = {r["candidate_id"] for r in read_jsonl(OUTPUT_PATH)}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

    torch.manual_seed(0)
    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for i, src in enumerate(test_records, start=1):
            text, arxiv_id = src["watermarked_text"], src["arxiv_id"]
            for lex, order in TARGETS:
                prompt = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_user_prompt(text, lex, order)}],
                    tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to("cuda")

                for model_name in ("base", "dpo"):
                    for rep in range(1, REPEATS + 1):
                        cid = f"{arxiv_id}_{model_name}_lex{lex}_ord{order}_rep{rep}"
                        if cid in done:
                            continue
                        ctx = model.disable_adapter() if model_name == "base" else nullcontext()
                        with ctx, torch.inference_mode():
                            ids = model.generate(
                                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True, temperature=0.7, top_p=0.9,
                                pad_token_id=tok.pad_token_id)
                        gen = tok.decode(ids[0, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip()
                        rec = {
                            "candidate_id": cid, "arxiv_id": arxiv_id, "model": model_name,
                            "lex_target": lex, "order_target": order, "repeat": rep,
                            "source_text": text, "paraphrased_text": gen,
                            "lex_actual": lexical_diversity(text, gen) if gen else None,
                            "order_actual": order_diversity(text, gen) if gen else None,
                            "watermark_detected": None, "watermark_detection_score": None,
                            "verify_error": None, "quality_score": None
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        done.add(cid)
                print(f"Testo {i}/{len(test_records)} done")

    del model, base
    free_gpu()

def score_all(test_records):
    records = read_jsonl(OUTPUT_PATH)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(SCORER_MODEL_NAME)
    lm = AutoModelForCausalLM.from_pretrained(SCORER_MODEL_NAME).to(dev).eval()
    rater = LLMTextRater(
        model=lm,
        tokenizer=tok,
        device=dev,
        cot=False,
        max_new_tokens=64,
        do_sample=False
    )
    tcfg = TransformersConfig(
        model=lm,
        tokenizer=tok,
        vocab_size=lm.config.vocab_size,
        device=dev,
        max_new_tokens=220,
        min_length=230,
        do_sample=True,
        no_repeat_ngram_size=4
    )
    wm = AutoWatermark.load(
        "KGW",
        algorithm_config=str(KGW_CONFIG_PATH),
        transformers_config=tcfg
    )

    for n, r in enumerate(records, start=1):
        if r["watermark_detected"] is not None or r["verify_error"]:
            continue
        try:
            d = wm.detect_watermark(r["paraphrased_text"])
            r["watermark_detected"] = bool(d["is_watermarked"])
            r["watermark_detection_score"] = float(d["score"])
            r["verify_error"] = False
        except Exception as e:
            r["verify_error"] = True
            print(f"Verify error {r['candidate_id']}: {e}")
        if r["quality_score"] is None and r["paraphrased_text"]:
            ref = ("Evaluate the quality of this paraphrase of an academic paper "
                   "conclusion. Consider whether it preserves the original meaning, "
                   "key findings and technical claims, and whether it is coherent, "
                   "grammatically correct and free from unsupported additions. "
                   "Do not reward lexical changes alone.\n\nOriginal conclusion:\n"
                   + r["source_text"])
            try:
                r["quality_score"] = max(0.0, min(1.0, float(
                    rater.analyze(text=r["paraphrased_text"], reference=ref))))
            except Exception as e:
                print(f"Quality error {r['candidate_id']}: {e}")
        if n % 20 == 0:
            write_jsonl_atomic(OUTPUT_PATH, records)

    src_det = []
    for s in test_records:
        try:
            src_det.append(bool(wm.detect_watermark(s["watermarked_text"])["is_watermarked"]))
        except Exception:
            pass
    write_jsonl_atomic(OUTPUT_PATH, records)
    return records, src_det

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def summarize(records, src_det):
    print(f"\nDetection on original texts: {sum(src_det)}/{len(src_det)}")
    ok = [r for r in records if r["verify_error"] is False]
    print(f"Valid records: {len(ok)}/{len(records)} (errors Verify: "
          f"{sum(1 for r in records if r['verify_error'])})")

    by = defaultdict(list)
    for r in ok:
        by[(r["model"], r["lex_target"], r["order_target"])].append(r)
        by[(r["model"], "TUTTI", "")].append(r)
    print(f"\n{'model':8} {'lex':>5} {'ord':>4} {'n':>4} {'detect%':>8} "
          f"{'lex_act':>8} {'ord_act':>8} {'qual':>6}")
    for k in sorted(by, key=lambda k: (str(k[1]) == "TUTTI", k[0], str(k[1]), str(k[2]))):
        rs = by[k]
        det = 100 * sum(r["watermark_detected"] for r in rs) / len(rs)
        print(f"{k[0]:8} {str(k[1]):>5} {str(k[2]):>4} {len(rs):>4} {det:8.1f} "
              f"{mean([r['lex_actual'] for r in rs]):8.1f} "
              f"{mean([r['order_actual'] for r in rs]):8.1f} "
              f"{mean([r['quality_score'] for r in rs]):6.2f}")

def main():
    test_records = read_jsonl(INPUT_PATH)[TRAIN_SIZE:]
    print(f"Testi di test: {len(test_records)}")
    generate_all(test_records)
    records, src_det = score_all(test_records)
    summarize(records, src_det)


if __name__ == "__main__":
    main()