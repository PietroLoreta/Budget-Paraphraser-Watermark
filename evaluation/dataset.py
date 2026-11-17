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

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json

import random
import re

from dataset.markmywords.generation_prompts import raw_prompts

from datasets import Dataset as HFDataset
from dataset.markmywords.generation_prompts_train import all_prompts as all_train_prompts

#Indicazione di lunghezza aggiunta ai prompt di training
LENGTH_HINT = " Respond in approximately 150 tokens."

#Cerca nel testo del prompt le parole di codice e li scartano
NON_PROSE = re.compile(r'\b(python|javascript|java|c\+\+|sql|function|implement|code|script|program|class|algorithm|regex|bash|shell|html|css|api)\b', re.I)
#Cerca nel testo del prompt le parole che danno un ruolo e li scartano
NON_PROSE_ROLE = re.compile(r'programmer|software|developer|engineer', re.I)

class BaseDataset:
    """Base class for dataset."""
    def __init__(self, max_samples: int = 200):
        """Initialize the dataset."""
        self.max_samples = max_samples
        self.prompts = []
        self.natural_texts = []
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass

class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""
    def __init__(self, data_source: str, max_samples: int = 200):
        """Initialize the C4 dataset."""
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])

class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""
    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """Initialize the WMT16 DE-EN dataset."""
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])

class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""
    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """Initialize the HumanEval dataset."""
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})

class MarkMyWordsDataset(BaseDataset):
    """Dataset class for MarkMyWords dataset."""
    def __init__(self, tokenizer, max_samples: int = 200, dataset="eval", seed: int = 42) -> None:
        """Initialize the MarkMyWords dataset."""
        super().__init__(max_samples)
        if dataset == "eval":
            self.max_samples = min(max_samples, len(raw_prompts))
            self.raw_prompts = raw_prompts[:self.max_samples]
        else:
            #Tiene solo i prompt in prosa (p è la richiesta e s è il ruolo)
            prose = [(p,s) for p, s in all_train_prompts
                     if not NON_PROSE.search(p) and not NON_PROSE_ROLE.search(s)]
            self.max_samples = min(max_samples, len(prose))
            #Fa una copia della lista da modificare
            shuffled = list(prose)
            #Crea un generatore di numeri casuali inizializzato con il seed e si mescola.
            #Si usa un generatore separato così nessun altro codice che usa random può alterare il risultato
            random.Random(seed).shuffle(shuffled)
            #Costruisce sempre la stessa lista di 400 coppie a ogni esecuzione
            self.raw_prompts = [(p + LENGTH_HINT, s) for p, s in shuffled[:self.max_samples]]
        self.prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                #Disattiva la modalità di ragionamento che consuma i token di generazione
                enable_thinking=False
            )
            for prompt, system_prompt in self.raw_prompts
        ]

class PaperConclusionDataset(BaseDataset):
    """Classe del dataset Paper Conclusion."""
    SYSTEM_PROMPT = "You are a helpful assistant that writes academic text."
    USER_PROMPT = (
        "Below is the title, abstract, and introduction of an academic paper. "
        "Write a conclusion summarizing its most important findings in "
        "approximately 200 tokens. Output only the conclusion.\n\n"
        "Title: {title}\n\nAbstract: {abstract}\n\nIntroduction: {introduction}"
    )

    def __init__(self, tokenizer, data_source: str, max_samples: int = 200,
                 seed: int = 42, max_intro_tokens: int = 1500) -> None:
        """Inizializza il dataset Paper Conclusion."""
        super().__init__(max_samples)
        self.tokenizer = tokenizer
        self.data_source = data_source
        #Seed dell'ordine dei paper: i risultati esistenti dipendono da quest'ordine
        self.seed = seed
        #Lunghezza massima dell'introduzione nel prompt in token
        self.max_intro_tokens = max_intro_tokens
        #Lista di prompt senza chat template (usata dal judge)
        self.raw_prompts = []
        #Lista di identificativi arXiv di ogni paper
        self.arxiv_ids = []
        self.load_data()

    def load_data(self):
        """Carica il file .arrow e prepara i prompt di tutti i paper."""
        data = HFDataset.from_file(self.data_source)

        #Posizioni dei paper nel file
        indices = list(range(len(data)))
        #Mescola le posizioni in modo casuale ma riproducibile con lo stesso seed
        random.Random(self.seed).shuffle(indices)
        #Tiene al massimo max_samples paper
        indices = indices[:self.max_samples]

        for idx in indices:
            #Il paper alla posizione idx del file .arrow
            item = data[idx]
            #Tronca l'introduzione a max_intro_tokens (niente token speciali di inizio/fine)
            intro_ids = self.tokenizer(item["introduction"], add_special_tokens=False, truncation=True, max_length=self.max_intro_tokens)["input_ids"]
            #Riconverte i token rimasti in testo ottenendo l'introduzione troncata
            introduction = self.tokenizer.decode(intro_ids, skip_special_tokens=True)
            #Riempie il modello di prompt con title, abstract e introduction
            user_prompt = self.USER_PROMPT.format(title=item['title'], abstract=item['abstract'], introduction=introduction)

            #Prompt "grezzo" come coppia utente/sistema. Lo usa il judge per sapere a quale richiesta risponde il testo
            self.raw_prompts.append((user_prompt, self.SYSTEM_PROMPT))
            #Prompt vero e proprio
            self.prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    #Restituisce una stringa (non token)
                    tokenize=False,
                    #Aggiunge in fondo l'inizio del turno dell'assistente
                    add_generation_prompt=True,
                    #Disattiva la modalità di ragionamento
                    enable_thinking=False
                )
            )
            #Conclusione scritta dagli autori del paper (testo umano)
            self.natural_texts.append(item["conclusion"])
            #Identificativo arXiv del paper
            self.arxiv_ids.append(item["arxiv_id"])

if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json', max_samples=100)
    d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl', max_samples=100)
    d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl', max_samples=100)
