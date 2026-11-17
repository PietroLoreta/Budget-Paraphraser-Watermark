import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from datasets import Dataset
#Modello instruct che viene addestrato a seguire istruzioni (come la parafrasi)
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = PROJECT_ROOT/"outputs"/"dipper_pilot"/ "KGW_preferences_eq2.jsonl"
OUTPUT_DIR = "/hdd10t/n333884/models/qwen2.5-3b-dpo-diversity"

SYSTEM_PROMPT = (
    "You are an expert copy-editor. Paraphrase the text you are given, preserving its "
    "meaning. Do not respond with a greeting or any extraneous information."
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
#Controlla se il tokenizer ha già un pad token (token vuoto)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

rows = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8") if l.strip()]

#Formatta il prompt nel formato di conversazione Qwen2.5
def to_chat_prompt(raw_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

ds = Dataset.from_list([
    #Costruisce un dizionario (prompt|chosen|rejected)
    {"prompt": to_chat_prompt(r["prompt"]), "chosen": r["chosen"], "rejected": r["rejected"]}
    for r in rows
]).shuffle(seed=42)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)

peft_config = LoraConfig(
    #Rango delle matrici di apprendimento
    r=16,
    #Fattore di scala applicato alla correzione
    lora_alpha=32,
    #Spegne il 5% delle connessioni per ridurre il rischio di sovradattamento
    lora_dropout=0.05,
    #Il modello genera testo token dopo token (come Qwen)
    task_type="CAUSAL_LM",
    #Elenco dei layer a cui attaccare gli adapter
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

#Carica gli iperparametri dell'addestramento DPO
args = DPOConfig(
    output_dir=OUTPUT_DIR,
    #Quanto il modello può allontanarsi dal modello di riferimento
    beta=0.1,
    #Dimensione di ogni passo di aggiornamento
    learning_rate=5e-5,
    #Quante volte il modello vede l'intero dataset
    num_train_epochs=3,
    #Ogni passo elabora una coppia alla volta
    per_device_train_batch_size=1,
    #Accumula i gradienti di 8 passi prima di aggiornare i pesi
    gradient_accumulation_steps=8,
    #Non conserva tutte le attivazioni intermedie, ma le ricalcola quando servono
    gradient_checkpointing=True,
    max_length=1024,
    bf16=True,
    #Scrive le metriche ogni 5 passi
    logging_steps=5,
    #Non salva checkpoint intermedi ma solo il modello finale alla fine
    save_strategy="no",
    #Non invia i log a servizi esterni
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    processing_class=tokenizer,
    peft_config=peft_config
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)