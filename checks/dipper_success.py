#Quanto spesso le parafrasi DIPPER rimuovono il watermark.

import argparse
import re

import pandas as pd

from evaluation.tools.diversity_metrics import lexical_diversity, order_diversity

#Nome delle sequenze DIPPER
BUDGET_PATTERN = re.compile(r"Dipper_L(\d+)_O(\d+)_r\d+")
#Fasce di diversità lessicale misurata
BINS = [0, 5, 10, 20, 30, 40, 60, 100]

def load(results_dir, watermark, key):
    """Righe delle parafrasi DIPPER di un file, con diversità misurata ed evasione."""
    d = pd.read_csv(f"{results_dir}/{watermark}_dipper_rated{key}.tsv", sep="\t")
    d = d[d.is_watermarked]
    #Testo originale di ogni id solo se il watermark è rilevato in partenza
    orig = d[(d.edit_sequence == "['none']") & d.watermark_detected].set_index("id").text
    rows = []
    for _, r in d[d.edit_sequence != "['none']"].iterrows():
        m = BUDGET_PATTERN.search(str(r.edit_sequence))
        if m is None or r.id not in orig.index:
            continue
        lex = lexical_diversity(orig[r.id], r.text)
        order = order_diversity(orig[r.id], r.text)
        if lex is None or order is None:
            continue
        rows.append({"watermark": watermark, "key": key, "id": r.id,
                     "budget": f"L{m.group(1)}_O{m.group(2)}",
                     "evaded": not r.watermark_detected, "L_measured": lex,
                     "O_measured": order})
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Share of DIPPER paraphrases that evade the watermark.")
    parser.add_argument("--watermarks", nargs="+", default=["KGW", "Unigram", "SynthID"])
    parser.add_argument("--keys", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--results_dir", default="evaluation/results/Qwen_Qwen3.5-0.8B")
    args = parser.parse_args()

    rows = []
    for wm in args.watermarks:
        for key in args.keys:
            rows += load(args.results_dir, wm, key)
    df = pd.DataFrame(rows)
    df.to_csv("checks/dipper_success.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n===== BY DIPPER BUDGET CODE =====")
    by_budget = df.groupby(["watermark", "budget"]).agg(
        paraphrases=("evaded", "size"), evasion=("evaded", "mean"),
        L_measured=("L_measured", "mean"), O_measured=("O_measured", "mean")
    )
    print(by_budget.round(2).to_string())

    print("\n===== BY MEASURED LEXICAL DIVERSITY =====")
    df["L_range"] = pd.cut(df.L_measured, BINS, include_lowest=True)
    rate = df.pivot_table(index="L_range", columns="watermark", values="evaded", aggfunc="mean", observed=False)
    count = df.pivot_table(index="L_range", columns="watermark", values="evaded", aggfunc="size", observed=False)
    print(rate.round(2).astype(str).add("(").add(count.fillna(0).astype(int).astype(str)).add(")").to_string())
    print("\nData for paraphrases saved in checks/dipper_success.csv")