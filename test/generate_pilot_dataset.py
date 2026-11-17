import sys
from pathlib import Path

#Identifica la cartella principale del progetto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

#Dataset e configurazione dell'algoritmo
DATASET_PATH = PROJECT_ROOT/"dataset"/"paper_conclusion"/"arxiv_hf.arrow"
CONFIG_PATH = PROJECT_ROOT/"config"/"KGW.json"

# File in cui salvare il risultato
OUTPUT_PATH = PROJECT_ROOT/"outputs"/"pilot"/"paper_conclusion"/"qwen3.5-0.8b"/"KGW_arxiv_hf_pilot_chat.jsonl"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    #Carica il dataset e seleziona un articolo
    print("Loading dataset...")
    dataset = Dataset.from_file(str(DATASET_PATH))

    item = dataset[24]
    print(f"Selected article: {item['arxiv_id']}")

    #Costruisce il prompt senza includere la conclusione di riferimento
    prompt = (
        "Below is the title, abstract, and introduction of an academic paper. "
        "Write a conclusion summarizing its most important findings in "
        "approximately 200 tokens. Output only the conclusion.\n\n"
        f"Title: {item['title']}\n\n"
        f"Abstract: {item['abstract']}\n\n"
        f"Introduction: {item['introduction']}"
    )

    print(f"Prompt length: {len(prompt)} characters")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B").to(device)
    model.eval()

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=64,
        do_sample=True,
        no_repeat_ngram_size=4,
    )

    print("Loading KGW...")
    myWatermark = AutoWatermark.load(
        "KGW",
        algorithm_config=str(CONFIG_PATH),
        transformers_config=transformers_config,
    )

    print("\nGenerating watermarked text...")
    watermarked_text = myWatermark.generate_watermarked_text(model_prompt)

    # Ricostruisce il prefisso nello stesso formato testuale
    # utilizzato da MarkLLM dopo la rimozione dei token speciali.
    encoded_prompt = tokenizer(
        model_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    decoded_prompt = tokenizer.batch_decode(
        encoded_prompt,
        skip_special_tokens=True,
    )[0]

    # Verifica che il testo generato inizi con il prefisso atteso.
    if not watermarked_text.startswith(decoded_prompt):
        print("Prefisso atteso:", repr(decoded_prompt[:200]))
        print("Testo restituito:", repr(watermarked_text[:200]))
        raise RuntimeError(
            "Generated text does not start with the decoded prompt."
        )

    # Estrae soltanto il completamento generato.
    completion = watermarked_text[len(decoded_prompt):].strip()

    #Salva il risultato in formato JSONL
    record = {
        "dataset": "paper_conclusion",
        "dataset_index": 24,
        "arxiv_id": item["arxiv_id"],
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "KGW",
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "max_new_tokens": 64,
        "prompt": prompt,
        "watermarked_text": completion,
        "generated_full": watermarked_text,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nGeneration complete!")
    print(f"Generated text length: {len(completion)} characters")
    print(f"Saved in:\n{OUTPUT_PATH}")

    print("\nGenerated text:")
    print(completion)


if __name__ == "__main__":
    main()