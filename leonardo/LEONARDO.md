# Esperimenti su Leonardo: un parafrasatore per watermark e per budget

Obiettivo: per 5 watermark (KGW, Unigram, SynthID, SIR, EXP), un parafrasatore per watermark e per budget,
valutato sul test Paper Conclusion contro tre riferimenti: DIPPER (ai budget che può raggiungere),
Qwen2.5-3B-Instruct non addestrato con la percentuale nel prompt, e la sostituzione di sinonimi.

Budget (L = diversità lessicale, O = diversità d'ordine, in %):

|Tipo|Budget|Peso λ della penalità|
|-|-|-|
|Solo lessicale|L2, L5, L10, L20, L30, L40, L60 (con O = 0)|8|
|Con riordino|L40\_O20|8|
|Con riordino|L60\_O20|1|
|Con riordino|L60\_O40|4|

I λ sono quelli della ricerca dei pesi per budget (validazione); per i budget nuovi si usa λ = 8.

Candidate per i dataset di preferenze: parafrasi DIPPER (6 budget × 2 ripetizioni) e sostituzione di
sinonimi (2, 5, 10, 15, 20% delle parole × 4 ripetizioni). Per KGW, Unigram e SynthID i testi watermarkati
e le parafrasi DIPPER sono già nell'archivio dei dati; per SIR ed EXP si generano da zero (fasi 1-3).

## 1\. Preparazione (una volta, sul nodo di login)

```bash
git clone https://github.com/PietroLoreta/Budget-Paraphraser-Watermark.git
cd Budget-Paraphraser-Watermark
#Dati già generati (testi watermarkati, parafrasi DIPPER con i voti, dataset di test)
tar xzf /percorso/leonardo\_data.tar.gz

python3 -m venv venv \&\& source venv/bin/activate     #versione di Python del server originale: Python 3.13.5
pip install -r requirements.txt

export HF\_HOME=/percorso/con/spazio/hf\_cache             #circa 80 GB di modelli
export NLTK\_DATA=$HF\_HOME/nltk\_data
hf auth login                                           #token con accesso a Llama-3.1-8B-Instruct
python3 leonardo/download\_models.py
```

## 2\. Ambiente di ogni job

Ogni job va lanciato **dalla cartella del progetto** con:

```bash
source venv/bin/activate
export PYTHONPATH=$(pwd)
export HF\_HOME=/percorso/con/spazio/hf\_cache
export NLTK\_DATA=$HF\_HOME/nltk\_data
export HF\_HUB\_OFFLINE=1 TRANSFORMERS\_OFFLINE=1        #i nodi di calcolo non scaricano niente
export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True
mkdir -p logs
```

## 3\. I job

In `leonardo/jobs/` c'è un file per fase, con **un comando per riga**. I job di una stessa fase sono
indipendenti e si possono lanciare in parallelo (per esempio come job array SLURM, la riga N al task N).
**Una fase può partire solo quando la precedente è finita.**

|Fase|File|Job|Contenuto|Risorse|
|-|-|-|-|-|
|1|`1\_generazione\_e\_sinonimi.txt`|17|SIR/EXP: testi watermarkati di training (3 chiavi) e di test. KGW/Unigram/SynthID: candidate con sinonimi|1 GPU|
|2|`2\_dipper\_sinonimi\_giudice.txt`|23|SIR/EXP: parafrasi DIPPER (training e test) e candidate con sinonimi. KGW/Unigram/SynthID: giudice sui sinonimi|1 GPU|
|3|`3\_giudice\_sir\_exp.txt`|14|SIR/EXP: giudice su parafrasi DIPPER e sinonimi|1 GPU|
|4|`4\_dataset\_preferenze.txt`|50|Dataset di preferenze (5 watermark × 10 budget)|solo CPU, circa 16 GB RAM|
|5|`5\_training.txt`|50|Training DPO + LoRA|1 GPU|
|6|`6\_test.txt`|60|Test: 50 modelli addestrati, Qwen non addestrato e sinonimi per ogni watermark|1 GPU|
|7|`7\_giudice\_test.txt`|60|Giudice sui risultati del test|1 GPU|

Durate indicative per job (stimate su una GPU condivisa da 46 GB): generazione 1-2 h; DIPPER 4-7 h sui dati
di training, meno di 1 h sul test; giudice 1-2 h sui dati di training, 10-30 min sul test; dataset 30-60 min;
training 30-60 min; test 10-30 min.

Se si vuole avere prima i risultati di KGW, Unigram e SynthID, le loro righe delle fasi 4-7 si possono lanciare
subito dopo la fase 2, senza aspettare la fase 3 (che riguarda solo SIR ed EXP).

## 4\. Cosa restituire

* `evaluation/results/Qwen\_Qwen3.5-0.8B/\*\_test\_\*\_rated15485863.tsv` (risultati sul test, pochi MB);
* `data/mix\_dipper\_\*/\*/Qwen/Qwen2.5-3B-Instruct\_pairs\_info.csv` (statistiche delle coppie);
* l'output dei job della fase 4 (righe `couples in ...`: coppie e testi scartati);
* i modelli in `models/per\_wm/` solo se servono ulteriori analisi (adapter LoRA, qualche decina di MB l'uno).

