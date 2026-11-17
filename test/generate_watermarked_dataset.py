import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

DATASET_PATH = PROJECT_ROOT/"dataset"/"paper_conclusion"/"arxiv_enhanced.arrow"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CONFIG_DIR = PROJECT_ROOT/"config"
ALGORITHMS = ["KGW", "SynthID", "Unigram"]

MAX_NEW_TOKENS = 220
MAX_INTRO_TOKENS = 20000

OUTPUT_DIR = PROJECT_ROOT/"outputs"/"watermarked_dataset"/"paper_conclusion"/"qwen3.5-0.8b"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
        min_length=230,
        do_sample=True,
        no_repeat_ngram_size=4
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = Dataset.from_file(str(DATASET_PATH))

    for algorithm_name in ALGORITHMS:
        print(f"Loading {algorithm_name}...")
        config_path = CONFIG_DIR/f"{algorithm_name}.json"

        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=str(config_path),
            transformers_config=transformers_config
        )

        output_path = OUTPUT_DIR/f"{algorithm_name}.jsonl"
        print(f"Output file: {output_path}")

        with output_path.open("w", encoding="utf-8") as output_files:
                for index, item in enumerate(dataset):

                    intro_ids = tokenizer(
                        item["introduction"],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=MAX_INTRO_TOKENS
                    )["input_ids"]

                    introduction = tokenizer.decode(
                        intro_ids,
                        skip_special_tokens=True
                    )

                    prompt = (
                        "Below is the title, abstract, and introduction of an academic paper. "
                        "Write a conclusion summarizing its most important findings in "
                        "approximately 200 tokens. Output only the conclusion.\n\n"
                        f"Title: {item['title']}\n\n"
                        f"Abstract: {item['abstract']}\n\n"
                        f"Introduction: {introduction}"
                    )

                    model_prompt = tokenizer.apply_chat_template(
                        #Comunica che stiamo preparando un messaggio dell'utente
                        [{"role": "user", "content": prompt}],
                        #La funzione restituisce una stringa formattata
                        tokenize=False,
                        #Aggiunge il marker che indica al modello dove iniziare la risposta
                        add_generation_prompt=True,
                        #Disabilita la modalità di ragionamento, ottenendo direttamente la conclusione
                        enable_thinking=False
                    )

                    print("\nGenerating watermarked text...")
                    watermarked_text = watermark.generate_watermarked_text(model_prompt)

                    encoded_prompt = tokenizer(
                        #Testo da convertire in token
                        model_prompt,
                        #Restituisce i dati come tensori PyTorch
                        return_tensors="pt",
                        #Aggiunge eventuali token speciali
                        add_special_tokens=True
                    )["input_ids"]

                    decoded_prompt = tokenizer.batch_decode(
                        encoded_prompt,
                        skip_special_tokens=True
                    )[0]

                    if not watermarked_text.startswith(decoded_prompt):
                        raise RuntimeError(
                            "Generated text does not start with the decoded prompt."
                        )
                    completion = watermarked_text[len(decoded_prompt):].strip()

                    record = {
                        "dataset": "paper_conclusion",
                        "dataset_file": "arxiv_enhanced.arrow",
                        "max_intro_tokens": MAX_INTRO_TOKENS,
                        "dataset_index": index,
                        "arxiv_id": item["arxiv_id"],
                        "model": "Qwen/Qwen3.5-0.8B",
                        "algorithm": algorithm_name,
                        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "watermarked_text": completion,
                    }

                    output_files.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )


if __name__ == "__main__":
    main()