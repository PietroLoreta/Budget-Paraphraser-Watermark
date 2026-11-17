# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==========================================
# test_method.py
# Description: Test the watermark algorithm
# ==========================================

import json
import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

#Carica tutte le righe ma prende solo la prima, interpretandola come JSON
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
item = json.loads(lines[0])
prompt = item['prompt']
natural_text = item['natural_text']

#Crea una funzione che riceve il nome dell'algoritmo
def test_algorithm(algorithm_name):
    #Evita di passare un nome che MarkLLM non conosce
    assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'Unbiased',
                              'UPV', 'TS', 'SynthID', 'EXP', 'EXPGumbel', 'EXPEdit', 'ITSEdit']

    #Controlla se è disponibile una GPU CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformers_config = TransformersConfig(
        #Carica il language model
        model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
        #Carica il token corrispondente
        tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
        vocab_size=50272,
        device=device,
        #Il modello può generare al massimo 200 nuovi token
        max_new_tokens=200,
        #Richiede una lunghezza minima nella generazione
        min_length=230,
        #La generazione non usa greedy decoding
        do_sample=True,
        #Impedisce la ripetizione dello stesso 4-gram
        no_repeat_ngram_size=4)

    #Carica l'algoritmo di watermark
    myWatermark = AutoWatermark.load(f'{algorithm_name}',
                                     algorithm_config=f'config/{algorithm_name}.json',
                                     transformers_config=transformers_config)

    #Generazione del testo con watermark
    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    print(watermarked_text)
    #Generazione del testo senza watermark
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

    #Il testo watermarkato dovrebbe essere rilevato
    detect_result = myWatermark.detect_watermark(watermarked_text)
    print(detect_result)
    #Il testo generato senza watermark dovrebbe essere molto meno compatibile
    detect_result = myWatermark.detect_watermark(unwatermarked_text)
    print(detect_result)
    #Il testo naturale del dataset subisce un ulteriore controllo
    detect_result = myWatermark.detect_watermark(natural_text)
    print(detect_result)


if __name__ == '__main__':
    test_algorithm('SynthID')