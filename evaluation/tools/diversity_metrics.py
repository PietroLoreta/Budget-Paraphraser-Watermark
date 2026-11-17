import math
import re

from collections import Counter, defaultdict, deque
from scipy.stats import kendalltau

def tokenize(text):
    """Divide un testo in token scartando la punteggiatura."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens

def lexical_diversity(original, paraphrase):
    """Diversità lessicale (0-100) tra un testo e la sua parafrasi."""
    original_tokens = tokenize(original)
    paraphrase_tokens = tokenize(paraphrase)

    if not original_tokens or not paraphrase_tokens:
        return None

    #Conta quante volte compare ogni parola
    common = Counter(original_tokens) & Counter(paraphrase_tokens)
    #Numero totale di parole in comune (ripetizioni comprese)
    overlap = sum(common.values())

    #Precision: quale frazione della parafrasi è fatta di parole dell'originale
    precision = overlap / len(paraphrase_tokens)
    #Recall: quale frazione dell'originale è rimasta nella parafrasi
    recall = overlap / len(original_tokens)

    #F1: media armonica di precision e recall, un unico valore tra 0 e 1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return 100 * (1 - f1)

def order_diversity(original, paraphrase):
    """Diversità d'ordine (0-100) tra un testo e la sua parafrasi."""
    original_tokens = tokenize(original)
    paraphrase_tokens = tokenize(paraphrase)

    #Per ogni parola dell'originale prende la coda delle posizioni in cui compare
    positions = defaultdict(deque)
    for i, token in enumerate(original_tokens):
        positions[token].append(i)

    #Abbina ogni parola della parafrasi a un'occorrenza della stessa parola nell'originale
    matched_positions = []
    for token in paraphrase_tokens:
        if positions[token]:
            matched_positions.append(positions[token].popleft())
    #Ottiene le posizioni originali delle parole in comune nell'ordine in cui compaiono nella parafrasi.
    #Se l'ordine è rimasto lo stesso, la lista è crescente

    #Con meno di due parole in comune non ci sono coppie da confrontare
    if len(matched_positions) < 2:
        return None

    #Kendall tau tra le posizioni originali e l'ordine della parafrasi che controlla se l'ordine è rimasto lo stesso.
    #+1 se tutte le coppie sono rimaste nello stesso ordine, -1 se sono tutte invertite e 0 se non c'è alcuna relazione
    tau = kendalltau(matched_positions, range(len(matched_positions))).statistic

    #Se non si riesce a calcolare tau è meglio restituire None
    if tau is None or not math.isfinite(float(tau)):
        return None

    return 50 * (1 - float(tau))