#Valuta un parafrasatore su un budget, sull'insieme di validazione. Per ogni parafrasi calcola
#evasione del watermark, qualità (voto del giudice), diversità lessicale e d'ordine misurate
#rispetto al testo originale; poi riassume per watermark e in totale, e aggiunge una riga a
#checks/weight_search.csv
#Lo stesso codice serve sia per i modelli addestrati che per DIPPER, così le misure sono
#confrontabili per costruzione
import argparse
import json
import os
import re

import pandas as pd

from evaluation.pipelines.pipeline_stages import ResponseLoader
from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity

parser = argparse.ArgumentParser()
parser.add_argument('label', help="config name, ex. L40_O20_l1 or DIPPER_L40_O20")
parser.add_argument('budget', help="budget target, ex. L40_O20")
parser.add_argument('files', nargs='+', help="TSV file to be rated")
parser.add_argument('--sequence', default=None, help="rate only paraphrases which contain this text")
args = parser.parse_args()

label, budget, files = args.label, args.budget, args.files
target_lex, target_order = (int(x) for x in re.findall(r'\d+', budget))
val_ids = set(json.load(open('data/val_ids.json')))

records = []
for f in files:
    watermark = os.path.basename(f).split('_')[0]
    rows = [r for r in ResponseLoader(f).load_all() if r.is_watermarked and r.id in val_ids]
    originals = {r.id: r for r in rows if r.edit_sequence == ['none']}
    for r in rows:
        #Parafrasi del modello valutato
        if r.edit_sequence == ['none'] or r.id not in originals or not originals[r.id].watermark_detected:
            continue
        if args.sequence is not None and args.sequence not in str(r.edit_sequence):
            continue
        orig = originals[r.id].text
        records.append({
            'watermark': watermark,
            'evaded': 0 if r.watermark_detected else 1,
            'quality': r.ratings.get('llm_judge'),
            'L_actual': lexical_diversity(orig, r.text),
            'O_actual': order_diversity(orig, r.text)
        })

df = pd.DataFrame(records).dropna()
df['L_err'] = (df.L_actual - target_lex).abs()
df['O_err'] = (df.O_actual - target_order).abs()

print(f'== {label} (target {budget}): {len(df)} rated paraphrases')
print(df.groupby('watermark')[['evaded', 'quality', 'L_actual', 'O_actual']].mean().round(2))

summary = {
    'config': label, 'target': budget, 'n': len(df),
    'evasion': round(df.evaded.mean(), 3),
    'quality': round(df.quality.mean(), 3),
    'L_actual': round(df.L_actual.mean(), 1),
    'O_actual': round(df.O_actual.mean(), 1),
    'L_err': round(df.L_err.mean(), 1),
    'O_err': round(df.O_err.mean(), 1),
}
print('\nSummary:', summary)

out = 'checks/weight_search.csv'
pd.DataFrame([summary]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
print(f'Added to {out}')