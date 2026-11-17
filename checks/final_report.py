import os
import re
import sys

import pandas as pd

from evaluation.pipelines.pipeline_stages import ResponseLoader
from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity

RES = sys.argv[1] if len(sys.argv) > 1 else 'evaluation/results/Qwen_Qwen3.5-0.8B'
KEY = 15485863
WATERMARKS = ['KGW', 'Unigram', 'SynthID']
BUDGETS = ['L20_O0', 'L40_O0', 'L60_O0', 'L40_O20', 'L60_O20', 'L60_O40']

def method_files(wm):
    """Nome del parafrasatore, file, budget."""
    yield 'DIPPER', f'{RES}/{wm}_test_dipper_rated{KEY}.tsv', None
    yield 'Qwen not trained', f'{RES}/{wm}_test_base_rated{KEY}.tsv', None
    yield 'Diaa et al.', f'{RES}/{wm}_test_diaa_rated{KEY}.tsv', '-'
    for b in BUDGETS:
        yield 'Your model', f'{RES}/{wm}_test_trained_{b}_rated{KEY}.tsv', b

records = []
for wm in WATERMARKS:
    for method, path, fixed_budget in method_files(wm):
        if not os.path.exists(path):
            continue
        rows = [r for r in ResponseLoader(path).load_all() if r.is_watermarked]
        originals = {r.id: r for r in rows if r.edit_sequence == ['none']}
        for r in rows:
            if r.edit_sequence == ['none'] or r.id not in originals or not originals[r.id].watermark_detected:
                continue
            if fixed_budget is None:
                m = re.search(r'L(\d+)_O(\d+)', str(r.edit_sequence))
                budget = f'L{m.group(1)}_O{m.group(2)}'
            else:
                budget = fixed_budget
            orig = originals[r.id].text
            records.append({
                'watermark': wm, 'method': method, 'budget': budget,
                'detected': int(r.watermark_detected),
                'quality': r.ratings.get('llm_judge'),
                'L_actual': lexical_diversity(orig, r.text),
                'O_actual': order_diversity(orig, r.text),
            })

df = pd.DataFrame(records)
order = ['-'] + BUDGETS
for metric, label in [('detected', 'DETECTION (rate of detected paraphrases'),
                      ('quality', 'QUALITY (average rating from the judge'),
                      ('L_actual', 'MEASURED LEXICAL DIVERSITY'),
                      ('O_actual', 'MEASURED ORDER DIVERSITY')]:
    print(f'\n===== {label} =====')
    for wm in WATERMARKS:
        t = df[df.watermark == wm].pivot_table(index='budget', columns='method', values=metric, aggfunc='mean')
        if t.empty:
            continue
        print(f'\n-- {wm} --')
        print(t.reindex([b for b in order if b in t.index]).round(2))

df.to_csv('checks/step4_results.csv', index=False)
print('\nSaved data for paraphrases in checks/step4_results.csv')