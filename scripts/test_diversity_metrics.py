import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tools.diversity_metrics import (
    lexical_diversity,
    order_diversity,
)

PILOT_PATH = PROJECT_ROOT / "outputs" / "dipper_pilot" / "KGW.jsonl"


def main():
    # 1. Test di controllo: testi identici
    text = "The model learns useful representations from data"

    lex_same = lexical_diversity(text, text)
    order_same = order_diversity(text, text)

    print("Test con testi identici")
    print(f"Lexical diversity: {lex_same}")
    print(f"Order diversity:   {order_same}")

    assert math.isclose(lex_same, 0.0, abs_tol=1e-9)
    assert math.isclose(order_same, 0.0, abs_tol=1e-9)

    # 2. Test di controllo: stesse parole in ordine inverso
    reversed_text = (
        "data from representations useful learns model the"
    )

    lex_reversed = lexical_diversity(text, reversed_text)
    order_reversed = order_diversity(text, reversed_text)

    print("\nTest con parole in ordine inverso")
    print(f"Lexical diversity: {lex_reversed}")
    print(f"Order diversity:   {order_reversed}")

    assert math.isclose(lex_reversed, 0.0, abs_tol=1e-9)
    assert math.isclose(order_reversed, 100.0, abs_tol=1e-9)

    # 3. Calcolo sulle parafrasi DIPPER già salvate
    if not PILOT_PATH.exists():
        raise FileNotFoundError(
            f"File non trovato: {PILOT_PATH}"
        )

    print("\nMetriche sulle parafrasi salvate")

    with PILOT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            original = record["source_text"]
            paraphrase = record["paraphrased_text"]

            lex_actual = lexical_diversity(
                original, paraphrase
            )
            order_actual = order_diversity(
                original, paraphrase
            )

            lex_target = record.get(
                "lex_diversity_target",
                record.get("lex_diversity")
            )
            order_target = record.get(
                "order_diversity_target",
                record.get("order_diversity")
            )

            print(
                f"\nTarget lex={lex_target}, "
                f"order={order_target}"
            )
            print(f"Actual lex:   {lex_actual}")
            print(f"Actual order: {order_actual}")

    print("\nTest completato correttamente.")


if __name__ == "__main__":
    main()