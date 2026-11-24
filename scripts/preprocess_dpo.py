import random
from collections import defaultdict
from typing import List

import os
import re
import argparse
import json

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.pipelines.pipeline_stages import ResponseLoader

from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity
from evaluation.tools.text_editor import build_budget_prompt, build_diaa_prompt

random.seed(42)

#Nomi delle sequenze candidate
CANDIDATE_PATTERN = re.compile(r"(Dipper_L\d+_O\d+|DipperPart_L\d+_p\d+|SynSub_q\d+)_r\d+")

def parse_budget(edit_sequence: List[str]) -> bool:
    """True se la riga è una parafrasi candidata (DIPPER, DIPPER parziale o sinonimi)."""
    return CANDIDATE_PATTERN.search(str(edit_sequence)) is not None

DIPPER_BUDGETS = [(20, 0), (40, 0), (60, 0), (40, 20), (60, 20), (60, 40)]  # (lexical, order)

#Pesi del punteggio di default: si usano quando non si specificano i pesi
ALPHA = 1.0
LAMBDA_L = 1.0
LAMBDA_O = 1.0
#Differenza minima di punteggio tra la parafrasi scelta e quella scartata
MIN_MARGIN = 0.1

#Insieme di validazione per la ricerca dei pesi
N_TRAIN_SETS = 400
VAL_SIZE = 40
VAL_SEED = 0
VAL_IDS = set(random.Random(VAL_SEED).sample(range(N_TRAIN_SETS), VAL_SIZE))

#Modello che ha generato i testi watermarked
PROVIDER = "Qwen/Qwen3.5-0.8B"
#Filtro di lunghezza
MIN_TOKENS, MAX_TOKENS = 128, 233 #10° e 90° percentile dei testi di Paper Conclusion
#Numero di coppie per testo
N_REJECTED = 3

def describe_candidate(original_text: str, paraphrase):
    """Descrive ogni parafrasi candidata."""
    lex = lexical_diversity(original_text, paraphrase.text)
    order = order_diversity(original_text, paraphrase.text)
    quality = paraphrase.ratings.get("llm_judge")
    if lex is None or order is None or quality is None:
        return None
    return {
        "response": paraphrase,
        "evaded": 0 if paraphrase.watermark_detected else 1,
        "quality": quality,
        "lex": lex,
        "order": order
    }

def score(c, target_lex, target_order, alpha=ALPHA, lambda_l=LAMBDA_L, lambda_o=LAMBDA_O):
    """Dà un voto a ogni parafrasi candidata."""
    return (c["evaded"] + alpha * c["quality"]
            - lambda_l * abs(c["lex"] - target_lex) / 100
            - lambda_o * abs(c["order"] - target_order) / 100)

def select_pairs(candidates, target_lex, target_order, n_rejected=N_REJECTED, min_margin=MIN_MARGIN, **weights):
    """Sceglie fino a n coppie (chosen/rejected) per un testo e un budget target."""
    #Servono almeno due candidate per formare una coppia
    if len(candidates) < 2:
        return []
    #Ordina le candidate dal punteggio più alto al più basso
    ranked = sorted(candidates, key=lambda c: score(c, target_lex, target_order, **weights), reverse=True)
    best = ranked[0]
    #Se nemmeno la migliore rompe il watermark si scarta
    if not best["evaded"]:
        return []
    best_score = score(best, target_lex, target_order, **weights)
    pairs = []
    #Parte dalla peggiore risalendo la classifica
    for worst in reversed(ranked[1:]):
        if len(pairs) == n_rejected:
            break
        if best_score - score(worst, target_lex, target_order, **weights) < min_margin:
            break
        pairs.append((best, worst))
    return pairs

def create_dpo_dataset(input_files: List[str], output_dir: str, model: str, target_lex: int, target_order: int,
                       prompt_style: str = "diaa", n_rejected: int = N_REJECTED,
                       min_margin: float = MIN_MARGIN, **weights) -> Dataset:
    """Dataset di preferenze per un budget target, con le coppie di tutti i file indicati."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    provider_tokenizer = AutoTokenizer.from_pretrained(PROVIDER)
    #Memorizza anche le informazioni delle coppie
    dpo_data, pairs_info = [], []
    #Misura quanti testi vengono scartati
    skipped = defaultdict(int)

    for input_group in input_files:
        # Gruppo = file dello stesso watermark e della stessa chiave (es. parafrasi DIPPER e sinonimi)
        input_group = [input_group] if isinstance(input_group, str) else input_group
        input_file = input_group[0]
        watermark = os.path.basename(input_file).split("_")[0]
        watermark_group = defaultdict(list)
        for f in input_group:
            for gen in ResponseLoader(f).load_all():
                if gen.is_watermarked:
                    watermark_group[gen.id].append(gen)

        # Un giro per ogni testo del file. watermark_group raggruppa le righe per id
        for _id in tqdm(watermark_group, desc=f"{input_file} L{target_lex}_0{target_order}"):
            if _id in VAL_IDS:
                skipped['validazione'] += 1
                continue
            id_group = watermark_group[_id]
            # Il testo originale, cioè la riga non attaccata
            no_attack = [g for g in id_group if g.edit_sequence == ['none']]
            # Si salta il testo se l'originale manca oppure se il watermark non viene rilevato in partenza
            if not no_attack or not no_attack[0].watermark_detected:
                skipped['original not detected'] += 1
                continue
            original_text = no_attack[0].text
            n_tokens = len(provider_tokenizer(original_text, add_special_tokens=False)["input_ids"])
            if not MIN_TOKENS <= n_tokens <= MAX_TOKENS:
                skipped['not in range length'] += 1
                continue

            #Tutte le parafrasi DIPPER del testo con le loro descrizioni
            candidates = [describe_candidate(original_text, g) for g in id_group if parse_budget(g.edit_sequence)]
            #Scarta le parafrasi per cui una misura non è disponibile
            candidates = [c for c in candidates if c is not None]
            pairs = select_pairs(candidates, target_lex, target_order, n_rejected, min_margin, **weights)
            if not pairs:
                skipped['no valid couple'] += 1
                continue

            prompt = (build_budget_prompt(tokenizer, original_text, target_lex, target_order)
                      if prompt_style == "budget" else build_diaa_prompt(tokenizer, original_text))
            for chosen, rejected in pairs:
                # Esempio di addestramento DPO
                dpo_data.append({'prompt': prompt,
                                 'chosen': chosen['response'].text,
                                 'rejected': rejected['response'].text})
                #Informazioni sulla coppia salvate a parte
                pairs_info.append({
                    'file': input_file, 'id': _id, 'watermark': watermark, 'n_tokens': n_tokens,
                    'target_lex': target_lex, 'target_order': target_order,
                    'chosen_seq': chosen['response'].edit_sequence[0],
                    'chosen_lex': chosen['lex'], 'chosen_order': chosen['order'],
                    'chosen_quality': chosen['quality'], 'chosen_evaded': chosen['evaded'],
                    'rejected_seq': rejected['response'].edit_sequence[0],
                    'rejected_lex': rejected['lex'], 'rejected_order': rejected['order'],
                    'rejected_quality': rejected['quality'], 'rejected_evaded': rejected['evaded']
                })

    dataset = Dataset.from_pandas(pd.DataFrame(dpo_data))
    dataset.save_to_disk(output_dir)
    #Salva le informazioni su tutte le coppie in un file CSV accanto al dataset
    pd.DataFrame(pairs_info).to_csv(f"{output_dir}_pairs_info.csv", index=False)
    print(f"L{target_lex}_0{target_order}: {len(dpo_data)} couples in {output_dir} | skipped texts: {dict(skipped)}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Watermark da includere: di default tutti e tre così il dataset è non adaptive
    parser.add_argument('--watermarks', nargs='+', default=['KGW', 'Unigram', 'SynthID'])
    #Chiavi di training da includere (la chiave di test qui non va mai usata)
    parser.add_argument('--keys', nargs='+', type=int, default=[10, 20, 30])
    #Cartella con i file delle parafrasi DIPPER valutate dal giudice
    parser.add_argument('--results_dir', default='evaluation/results/Qwen_Qwen3.5-0.8B')
    #Budget per cui costruire i dataset (di default vengono usati tutti e sei)
    parser.add_argument('--budgets', nargs='+', default=None)
    #Prompt salvato nelle coppie: 'diaa' (usato dal modello addestrato) o 'budget'
    parser.add_argument('--prompt_style', choices=['diaa', 'budget'], default='diaa')
    #Pesi del punteggio
    parser.add_argument('--alpha', type=float, default=ALPHA)
    parser.add_argument('--lambda_l', type=float, default=LAMBDA_L)
    parser.add_argument('--lambda_o', type=float, default=LAMBDA_O)
    #Numero di parafrasi "rejected" per testo
    parser.add_argument('--n_rejected', type=int, default=N_REJECTED)
    #Suffisso del nome della cartella di output
    parser.add_argument('--tag', type=str, default='')
    # Suffissi dei file di candidate da unire per ogni watermark e chiave
    parser.add_argument('--input_suffixes', nargs='+', default=['_dipper_rated'])
    args = parser.parse_args()

    #Salva gli id di validazione. Questi testi sono esclusi dai dataset e usati in valutazione
    with open("data/val_ids.json", "w") as f:
        json.dump(sorted(VAL_IDS), f)

    #Un file di parafrasi DIPPER per ogni combinazione watermark x chiave (3 x 3 = 9)
    input_files = [[f"{args.results_dir}/{wm}{suff}{key}.tsv" for suff in args.input_suffixes]
                   for wm in args.watermarks for key in args.keys]
    #Budget da elaborare, trasformati da "L40,O20" a (40, 20)
    budgets = (DIPPER_BUDGETS if args.budgets is None
               else [tuple(int(x) for x in re.findall(r'\d+', b)) for b in args.budgets])
    #Modello che verrà addestrato
    model = "Qwen/Qwen2.5-3B-Instruct"

    #Un dataset per ogni budget, nella sua cartella
    for target_lex, target_order in budgets:
        output_dir = f"data/mix_dipper{args.tag}/L{target_lex}_O{target_order}/{model}"
        create_dpo_dataset(input_files, output_dir, model, target_lex, target_order,
                           prompt_style=args.prompt_style, n_rejected=args.n_rejected,
                           alpha=args.alpha, lambda_l=args.lambda_l, lambda_o=args.lambda_o)