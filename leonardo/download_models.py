"""Scarica in anticipo modelli e dati NLTK, per i nodi di calcolo senza accesso a internet.

Da lanciare sul nodo di login, dalla cartella del progetto, con HF_HOME e NLTK_DATA già impostati
(vedi leonardo/LEONARDO.md). Per Llama-3.1-8B-Instruct serve un token Hugging Face con accesso
approvato al modello: prima `hf auth login` (oppure la variabile HF_TOKEN).

Uso: python3 leonardo/download_models.py
"""
import nltk
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3.5-0.8B",                                  #Provider dei testi watermarkati
    "Qwen/Qwen2.5-3B-Instruct",                           #Parafrasatore addestrato e baseline non addestrata
    "kalpeshk2011/dipper-paraphraser-xxl",                #DIPPER
    "google/t5-v1_1-xxl",                                 #Tokenizer di DIPPER
    "bert-large-uncased",                                 #Sostituzione di sinonimi
    "perceptiveshawty/compositional-bert-large-uncased",  #Embedding del watermark SIR
    "meta-llama/Llama-3.1-8B-Instruct",                   #Giudice della qualità (accesso controllato)
]

for name in MODELS:
    print(f"== {name}")
    snapshot_download(name)

#Divisione in frasi (punkt) e sinonimi (wordnet)
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg)
print("== Fatto")
