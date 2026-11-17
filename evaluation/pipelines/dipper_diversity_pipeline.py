import sys
import gc
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,T5ForConditionalGeneration,T5Tokenizer
from evaluation.tools.text_editor import DipperParaphraser
from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity
from evaluation.tools.text_quality_analyzer import LLMTextRater
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

INPUT_PATH = PROJECT_ROOT/"outputs"/"watermarked_dataset"/"paper_conclusion"/"qwen3.5-0.8b"/"KGW.jsonl"
OUTPUT_PATH = PROJECT_ROOT/"outputs"/"dipper_pilot"/"KGW_candidates.jsonl"
PREFERENCES_PATH = PROJECT_ROOT/"outputs"/"dipper_pilot"/"KGW_preferences_eq2.jsonl"
AUDIT_PATH = PROJECT_ROOT/"outputs"/"dipper_pilot"/"KGW_preferences_eq2_audit.jsonl"

TOKENIZER_PATH = "google/t5-v1_1-xxl"
MODEL_PATH = "/hdd10t/n333884/models/dipper-paraphraser-xxl"
SCORER_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
KGW_CONFIG_PATH = PROJECT_ROOT/"config"/"KGW.json"

MAX_SOURCE_RECORDS = 50
DIVERSITY_CONFIGS = [(20, 0),(40, 0),(60, 0),(40, 20),(60, 20),(60, 40)]
REPEATS_PER_TARGET = 5

QUALITY_THRESHOLD = 0.70
LAMBDA_LEX = 1.0
LAMBDA_ORDER = 1.0
CHECKPOINT_EVERY = 10
MAX_PENALTY_LEX = 0.0225
MAX_PENALTY_ORDER = 0.0225

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl_atomic(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
    tmp_path.replace(path)

#Costruisce l'insieme di tutti gli ID dei candidati che ci aspettiamo di generare
def expected_candidate_ids(source_records):
    return {
        f"{source['arxiv_id']}_lex{lex_target}_ord{order_target}_rep{repeat}"
        #Per ogni testo originale presente
        for source in source_records
        #Per ogni tupla con due valori
        for lex_target, order_target in DIVERSITY_CONFIGS
        #Per ogni ripetizione
        for repeat in range(1, REPEATS_PER_TARGET + 1)
    }

def generate_candidates(source_records):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()

    if OUTPUT_PATH.exists():
        existing = read_jsonl(OUTPUT_PATH)
        for index, record in enumerate(existing, start=1):
            #Controlla che ogni record abbia un identificatore
            if "candidate_id" not in record:
                raise RuntimeError(f"Record {index} in {OUTPUT_PATH} has no candidate_id")
            #Controlla che non ci siano identificatori duplicati
            candidate_id = record["candidate_id"]
            if candidate_id in completed_ids:
                raise RuntimeError(f"Duplicate candidate_id in file: {candidate_id}")
            completed_ids.add(candidate_id)
        print(f"Already saved records: {len(completed_ids)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("Loading DIPPER tokenizer...")
    dipper_tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)

    print("Loading DIPPER model...")
    dipper_model = T5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        #Decide di collocare i diversi componenti del modello in automatico
        device_map="auto",
        max_memory={0: "29GiB"},
    )
    dipper_model.eval()

    new_count = 0
    try:
        with OUTPUT_PATH.open("a", encoding="utf-8") as out:
            for source_idx, source_record in enumerate(source_records, start=1):
                source_text = source_record["watermarked_text"]
                arxiv_id = source_record["arxiv_id"]

                for lex_target, order_target in DIVERSITY_CONFIGS:
                    print(f"\nText {source_idx}/{len(source_records)} | lex={lex_target} | order={order_target}")

                    paraphraser = DipperParaphraser(
                        tokenizer=dipper_tokenizer,
                        model=dipper_model,
                        device="cuda",
                        lex_diversity=lex_target,
                        order_diversity=order_target,
                        #Indica quante frasi elaborare alla volta
                        sent_interval=3,
                        #Indica un limite alla lunghezza generata
                        max_length=512,
                        do_sample=True,
                        #Indica la probabilità minima dei token che possono essere scelti
                        top_p=0.75,
                        #Indica il numero di token da considerare (disabilitato)
                        top_k=0,
                    )

                    for repeat in range(1, REPEATS_PER_TARGET + 1):
                        candidate_id = f"{arxiv_id}_lex{lex_target}"f"_ord{order_target}_rep{repeat}"
                        if candidate_id in completed_ids:
                            print(f"Already complete, skipping: {candidate_id}")
                            continue

                        paraphrased_text = ""

                        candidate = paraphraser.edit(source_text, reference="")
                        if not candidate or not candidate.strip():
                            print(f"WARNING: empty output for {candidate_id}")
                            continue

                        paraphrased_text = candidate.strip()

                        record = {
                            "candidate_id": candidate_id,
                            "arxiv_id": arxiv_id,
                            "source_algorithm": "KGW",
                            #Indica quale ripetizione della stessa configurazione
                            "repeat": repeat,
                            "lex_diversity_target": lex_target,
                            "order_diversity_target": order_target,
                            "lex_diversity_actual": lexical_diversity(source_text, paraphrased_text),
                            "order_diversity_actual": order_diversity(source_text, paraphrased_text),
                            "quality_score": None,
                            "watermark_detected": None,
                            "watermark_detection_score": None,
                            "source_text": source_text,
                            "paraphrased_text": paraphrased_text,
                        }

                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()

                        completed_ids.add(candidate_id)
                        new_count += 1
                        print(f"Actual lex={record['lex_diversity_actual']}, order={record['order_diversity_actual']}")
    finally:
        try:
            #Elimina il parafrasatore
            del paraphraser
        except UnboundLocalError:
            pass
        #Elimina il modello e il tokenizer
        del dipper_model
        del dipper_tokenizer
        #Recupera la memoria inutilizzata (garbage collector)
        gc.collect()
        #Libera la cache CUDA inutilizzata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"New candidates generated: {new_count}")
    print(f"Candidate file: {OUTPUT_PATH}")

def score_and_verify_candidates():
    records = read_jsonl(OUTPUT_PATH)
    needs_work = any(
        #Controlla se il punteggio di qualità non è ancora stato calcolato
        record.get("quality_score") is None
        #Controlla se la verifica del watermark non è ancora stata eseguita
        or record.get("watermark_detected") is None
        or record.get("source_detected") is None
        #Esamina uno per uno tutti i candidati
        for record in records
    )

    if needs_work:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading scorer/verifier model: {SCORER_MODEL_NAME} on {device}...")

        qwen_tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL_NAME)

        qwen_model = AutoModelForCausalLM.from_pretrained(SCORER_MODEL_NAME).to(device)
        qwen_model.eval()

        rater = LLMTextRater(
            model=qwen_model,
            tokenizer=qwen_tokenizer,
            device=device,
            #Indica se il valutatore deve essere configurato (Chain of Thought)
            cot=False,
            max_new_tokens=64,
            do_sample=False
        )

        transformers_config = TransformersConfig(
            model=qwen_model,
            tokenizer=qwen_tokenizer,
            vocab_size=qwen_model.config.vocab_size,
            device=device,
            max_new_tokens=220,
            min_length=230,
            do_sample=True,
            no_repeat_ngram_size=4
        )
        watermark = AutoWatermark.load(
            "KGW",
            algorithm_config=str(KGW_CONFIG_PATH),
            transformers_config=transformers_config
        )

        processed_since_checkpoint = 0
        try:
            source_cache = {}

            for index, record in enumerate(records, start=1):
                quality_missing = record.get("quality_score") is None
                verify_missing = record.get("watermark_detected") is None
                if not quality_missing and not verify_missing and record.get("source_detected") is not None:
                    continue

                candidate_id = record.get("candidate_id", f"record_{index}")
                print(f"Scoring/verifying {index}/{len(records)}: {candidate_id}")

                if quality_missing:
                    reference = (
                        "Evaluate the quality of this paraphrase of an academic paper "
                        "conclusion. Consider whether it preserves the original meaning, "
                        "key findings and technical claims, and whether it is coherent, "
                        "grammatically correct and free from unsupported additions. "
                        "Do not reward lexical changes alone.\n\n"
                        "Original conclusion:\n" + record["source_text"]
                    )
                    try:
                        quality = float(
                            rater.analyze(
                                text=record["paraphrased_text"],
                                reference=reference
                            )
                        )
                        record["quality_score"] = max(0.0, min(1.0, quality))
                    except Exception as exc:
                        print(f"ERROR quality for {candidate_id}: {exc}")

                if verify_missing:
                    try:
                        detection = watermark.detect_watermark(record["paraphrased_text"])
                        record["watermark_detected"] = bool(detection["is_watermarked"])
                        record["watermark_detection_score"] = float(detection["score"])
                    except Exception as exc:
                        print(f"ERROR KGW verify for {candidate_id}: {exc}")
                        record["watermark_detected"] = False
                        record["watermark_detection_score"] = None
                        record["verify_error"] = True

                if record.get("source_detected") is None:
                    src = record["source_text"]
                    if src not in source_cache:
                        d = watermark.detect_watermark(src)
                        source_cache[src] = bool(d["is_watermarked"])
                    record["source_detected"] = source_cache[src]

                processed_since_checkpoint += 1
                #Salva periodicamente i risultati per evitare di perdere troppo lavoro
                if processed_since_checkpoint >= CHECKPOINT_EVERY:
                    write_jsonl_atomic(OUTPUT_PATH, records)
                    processed_since_checkpoint = 0
        finally:
            write_jsonl_atomic(OUTPUT_PATH, records)
            del watermark
            del transformers_config
            del rater
            del qwen_model
            del qwen_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    records = read_jsonl(OUTPUT_PATH)
    incomplete = [
        r.get("candidate_id", "unknown")
        for r in records
        if r.get("quality_score") is None
        or r.get("watermark_detected") is None
        or r.get("source_detected") is None
    ]
    if incomplete:
        raise RuntimeError(f"Quality/verification incomplete for {len(incomplete)} records. ")

    print(f"Quality and KGW verification complete for {len(records)} candidates.")
    return records

#Valuta quanto è buona ogni parafrasi secondo i nostri criteri
def add_objective_scores(records):
    scored = []
    for original in records:
        #Crea una copia del dizionario original
        record = dict(original)
        lex_actual = record.get("lex_diversity_actual")
        order_actual = record.get("order_diversity_actual")

        record["evasion_reward"] = (0.0 if bool(record["watermark_detected"]) else 1.0)
        record["quality_pass"] = (float(record["quality_score"]) >= QUALITY_THRESHOLD)
        record["watermark_evaded"] = not bool(record["watermark_detected"])

        if lex_actual is None or order_actual is None:
            record["penalty_lex"] = None
            record["penalty_order"] = None
            record["objective_score"] = None
            record["score_exclusion_reason"] = "undefined diversity metric"
        else:
            penalty_lex = ((float(lex_actual) - float(record["lex_diversity_target"])) / 100.0) ** 2
            penalty_order = ((float(order_actual) - float(record["order_diversity_target"])) / 100.0) ** 2
            reward = (
                record["evasion_reward"]
                + float(record["quality_score"])
                - LAMBDA_LEX * penalty_lex
                - LAMBDA_ORDER * penalty_order
            )
            record["penalty_lex"] = penalty_lex
            record["penalty_order"] = penalty_order
            record["objective_score"] = reward
            record["score_exclusion_reason"] = None

        scored.append(record)
    write_jsonl_atomic(OUTPUT_PATH, scored)
    return scored

#Trasforma i candidati valutati in un dataset di preferenze (chosen/rejected)
def build_preference_pairs(records):
    groups = defaultdict(list)
    for record in records:
        #Raggruppa i candidati per articolo originale e diversità
        key = (
            record["arxiv_id"],
            record["lex_diversity_target"],
            record["order_diversity_target"],
        )
        groups[key].append(record)

    preferences = []
    audit = []
    skipped_no_positive = 0
    skipped_bad_metrics = 0
    skipped_source_undetected = 0

    #Elabora i gruppi in un ordine deterministico
    for (arxiv_id, lex_target, order_target), group in sorted(groups.items(), key=lambda item: str(item[0])):
        #Seleziona i candidati con uno score valido
        valid = [r for r in group if r.get("objective_score") is not None]

        if not group[0]["source_detected"]:
            skipped_source_undetected += 1
            continue

        if not valid:
            skipped_bad_metrics += 1
            continue

        positives = [
            r for r in valid
            if not bool(r["watermark_detected"])
            and float(r["quality_score"]) >= QUALITY_THRESHOLD
            and r["penalty_lex"] <= MAX_PENALTY_LEX
            and r["penalty_order"] <= MAX_PENALTY_ORDER
            and not r.get("verify_error")
        ]

        if not positives:
            skipped_no_positive += 1
            continue
        positive_ids = {r["candidate_id"] for r in positives}

        negatives = [r for r in valid if r["candidate_id"] not in positive_ids]

        for chosen in positives:
            #Sceglie il primo score negativo più alto sotto quello scelto
            lower_scoring_negatives = [r for r in negatives if float(r["objective_score"]) < float(chosen["objective_score"])]

            if lower_scoring_negatives:
                rejected = min(lower_scoring_negatives, key=lambda r: float(r["objective_score"]))
                rejected_text = rejected["paraphrased_text"]
                rejected_id = rejected["candidate_id"]
                rejected_score = float(rejected["objective_score"])
                rejected_is_original = False
            else:
                rejected = None
                rejected_text = chosen["source_text"]
                rejected_id = "original_watermarked_source"
                rejected_score = None
                rejected_is_original = True

            prompt = (
                "Paraphrase the following academic conclusion while preserving its "
                "scientific meaning, findings, and technical claims. Follow the "
                "requested diversity targets as closely as possible.\n"
                f"Target lexical diversity: {lex_target}/100.\n"
                f"Target order diversity: {order_target}/100.\n\n"
                "[[START OF TEXT]]\n"
                f"{chosen['source_text']}\n"
                "[[END OF TEXT]]\n"
                "[[START OF PARAPHRASE]]\n"
            )

            preferences.append({
                "prompt": prompt,
                "chosen": chosen["paraphrased_text"],
                "rejected": rejected_text,
            })
            audit.append({
                "arxiv_id": arxiv_id,
                "lex_diversity_target": lex_target,
                "order_diversity_target": order_target,
                "chosen_candidate_id": chosen["candidate_id"],
                "chosen_score": float(chosen["objective_score"]),
                "chosen_quality_score": float(chosen["quality_score"]),
                "chosen_watermark_detected": bool(chosen["watermark_detected"]),
                "chosen_penalty_lex": chosen["penalty_lex"],
                "chosen_penalty_order": chosen["penalty_order"],
                "rejected_candidate_id": rejected_id,
                "rejected_score": rejected_score,
                "rejected_is_original_source": rejected_is_original,
                "rejected_quality_score": (float(rejected["quality_score"]) if rejected is not None else None),
                "rejected_watermark_detected": (bool(rejected["watermark_detected"]) if rejected is not None else True),
                "lambda_lex": LAMBDA_LEX,
                "lambda_order": LAMBDA_ORDER,
                "quality_threshold": QUALITY_THRESHOLD,
                "num_candidates_in_group": len(group),
            })

    write_jsonl_atomic(PREFERENCES_PATH, preferences)
    write_jsonl_atomic(AUDIT_PATH, audit)

    print("\nPreference curation summary")
    print(f"Candidate records scored: {len(records)}")
    print(f"Original/target groups: {len(groups)}")
    print(f"Preference pairs created: {len(preferences)}")
    print(f"Groups skipped (no eligible positive): {skipped_no_positive}")
    print(f"Groups skipped (undefined diversity metrics): {skipped_bad_metrics}")
    print(f"Groups skipped (source not detected): {skipped_source_undetected}")
    print(f"Candidates with scores: {OUTPUT_PATH}")
    print(f"DPO dataset: {PREFERENCES_PATH}")
    print(f"Selection audit: {AUDIT_PATH}")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    source_records = read_jsonl(INPUT_PATH)[:MAX_SOURCE_RECORDS]
    if len(source_records) != MAX_SOURCE_RECORDS:
        raise RuntimeError(f"Expected {MAX_SOURCE_RECORDS} source records, found {len(source_records)}.")

    source_ids = [record["arxiv_id"] for record in source_records]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate arxiv_id values found in source records.")

    expected_ids = expected_candidate_ids(source_records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source records: {len(source_records)}")
    print(f"Diversity configurations: {DIVERSITY_CONFIGS}")
    print(f"Repeats per configuration: {REPEATS_PER_TARGET}")
    print(f"Expected candidate count: {len(expected_ids)}")

    generate_candidates(source_records)

    candidates = read_jsonl(OUTPUT_PATH)
    candidate_ids = {record["candidate_id"] for record in candidates}
    missing_ids = expected_ids - candidate_ids
    unexpected_ids = candidate_ids - expected_ids
    if missing_ids or unexpected_ids:
        print(f"Missing candidate IDs: {len(missing_ids)}")
        if missing_ids:
            print("Examples:", sorted(missing_ids)[:10])
        if unexpected_ids:
            print(f"Unexpected candidate IDs: {len(unexpected_ids)}")
            print("Examples:", sorted(unexpected_ids)[:10])
        raise RuntimeError(
            "Candidate generation is incomplete or OUTPUT_PATH contains records "
            "from a different configuration. Fix/resume generation before scoring."
        )

    candidates = score_and_verify_candidates()
    candidates = add_objective_scores(candidates)
    build_preference_pairs(candidates)


if __name__ == "__main__":
    main()