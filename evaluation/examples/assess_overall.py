import json
import os
from typing import Dict, List

import gc
import random
import torch

from evaluation.dataset import MarkMyWordsDataset, PaperConclusionDataset
from evaluation.pipelines.robustness import WatermarkRobustnessPipeline, PipelineConfig
from evaluation.tools.text_editor import TruncatePromptTextEditor, LLMParaphraser, \
    GPTParaphraser, ContextAwareSynonymSubstitution, DipperParaphraser
from evaluation.tools.text_quality_analyzer import GPTTextRater, LLMTextRater
from evaluation.utils import ModelLoader, ModelConfig, _load_tokenizer
from utils.transformers_config import TransformersConfig

from evaluation.tools.text_editor import BudgetLLMParaphraser, TrimToLastSentenceTextEditor, PartialDipperParaphraser, SparseSynonymSubstitution
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, BertTokenizer, BertForMaskedLM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
NUM_PARAPHRASES = 8
MAX_NEW_TOKENS = 512
#Budget di diversità con cui viene interrogato DIPPER
DIPPER_BUDGETS = [(20, 0), (40, 0), (60, 0), (40, 20), (60, 20), (60, 40)]  # (lexical, order)
#Quante parafrasi DIPPER generare per ogni testo e per ogni budget
DIPPER_REPEATS = 2

#Budget della baseline Qwen non addestrato
QWEN_BUDGETS = [(2, 0), (5, 0), (10, 0), (20, 0), (30, 0), (40, 0), (60, 0), (40, 20), (60, 20), (60, 40)]

#Parafrasi parziali per i budget piccoli
PARTIAL_FRACTIONS = []
PARTIAL_LEX = 40
#Ripetizioni per quota
PARTIAL_REPEATS = 4

#Sostituzione di sinonimi sparsa per i budget piccoli
SYNONYM_RATIOS = [0.02, 0.05, 0.1, 0.15, 0.2]
SYNONYM_REPEATS = 4

class RobustnessPipelineFactory:
    """Factory for creating watermark robustness evaluation pipelines."""
    def __init__(self, devices: List[str]):
        self.devices = devices

    def create_pipeline(self,
                        dataset: MarkMyWordsDataset,
                        mode: str = 'end_to_end',
                        filename: str = None,
                        paraphraser: str = None,
                        #La pipeline gira con un giudice Llama-3.1-8B
                        no_judge: bool = False,
                        #Testi valutati insieme dal giudice
                        judge_batch_size: int = 16
                        ) -> 'WatermarkRobustnessPipeline':
        """Create a pipeline with standard configuration."""
        edit_sequences = {}
        if mode == 'gpt_judge':
            rating_metrics = {"gpt_judge": GPTTextRater(openai_model='gpt-4o-mini', cot=True)}
            batch_size = 32
        elif mode == 're_evaluate':
            rating_metrics = RobustnessPipelineFactory._create_rating_metrics(self.devices[-3:])
            #L'originale (128) è pensato per un server multi-GPU
            batch_size = judge_batch_size
        else:
            #In DIPPER il giudice non viene caricato: DIPPER e Llama-8B insieme non entrano nella GPU condivisa.
            #La qualità viene calcolata in un secondo passaggio quando DIPPER non è più in memoria
            if mode in ['dipper_paraphrase', 'dipper_partial_paraphrase', 'budget_paraphrase', 'llm_paraphrase'] or no_judge:
                rating_metrics = {}
            else:
                rating_metrics = RobustnessPipelineFactory._create_rating_metrics(self.devices)
            batch_size = 16
            devices = self.devices
            if mode == 'generate_only':
                #Toglie il prompt dall'output e taglia il testo all'ultima frase completa
                edit_sequences = {'none': [TruncatePromptTextEditor(), TrimToLastSentenceTextEditor()]}
            elif mode == 'end_to_end':
                devices = devices[1:]
                edit_sequences = RobustnessPipelineFactory._create_edit_sequences(devices)
            elif mode == 'add_edits':
                edit_sequences = RobustnessPipelineFactory._create_extra_edit_sequences(devices)
            elif mode == 'base_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_base_paraphrase_sequences(devices)
            elif mode == 'ensemble_paraphrase':
                model_names = ["models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", 'models/dpo_qwen_3_exponential']
                devices_per_model = len(devices) // len(model_names)
                # lambda_weights = [0, 0.2, 0.4, 0.6, 0.8, 1]
                lambda_weights = [0, 0.5, 1]
                edit_sequences = {}
                batch_size = 32
                for i, model_name in enumerate(model_names):
                    edit_sequences.update(
                        RobustnessPipelineFactory._create_weighted_paraphraser_sequences(
                            model_name, devices[i * devices_per_model:(i + 1) * devices_per_model], lambda_weights
                        )
                    )
            #Attacca con DIPPER i testi watermarkati già generati a tutti i budget di diversità
            elif mode == 'dipper_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_dipper_sequences(devices)
            #Parafrasi parziali per i budget piccoli
            elif mode == 'dipper_partial_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_partial_dipper_sequences(devices)
            elif mode == 'budget_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_budget_sequences(paraphraser, devices)
            elif mode == 'llm_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_llm_sequences(paraphraser, devices)
            else:
                raise ValueError(f"Invalid mode {mode}.")
        config = PipelineConfig(dataset=dataset,
                                edit_sequences=edit_sequences,
                                rating_metrics=rating_metrics,
                                filename=filename,
                                batch_size=batch_size,
                                max_batch_size=min(8, batch_size))
        return WatermarkRobustnessPipeline(config)

    @staticmethod
    def _create_extra_edit_sequences(devices: List[str]) -> Dict:
        """Create extra set of text editing sequences."""
        bert = ModelLoader.from_name('bert-large-uncased', device=devices[0])
        paraphraser_models = [
            # ("models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", devices[0]),
            # ("models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct", devices[0]),
            ("meta-llama/Llama-3.2-3B-Instruct", devices[0]),
        ]
        sequences = {'Word-S(Context)': [ContextAwareSynonymSubstitution(ratio=0.5, **bert)],}
        for model, device in paraphraser_models:
            paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device)
            # no prompt truncate, because "none" is already truncated
            sequences[f'Paraphrase({model})'] = [paraphraser]
        return sequences

    @staticmethod
    def _create_edit_sequences(devices: List[str]) -> Dict:
        """Create standard set of text editing sequences."""
        base_editor = TruncatePromptTextEditor()
        sequences = {
            'none': [base_editor],
            # 'Word-D': [base_editor, WordDeletion(ratio=0.3)],
            # 'Word-S': [base_editor, SynonymSubstitution(ratio=0.5)],
            'Paraphrase(GPT3.5)': [GPTParaphraser('gpt-3.5-turbo')],
            'Paraphrase(GPT4o)': [GPTParaphraser('gpt-4o')]
        }

        # Add paraphrasers
        # paraphraser_models = [
        #     ("models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", devices[1]),
        #     ("models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct", devices[3]),
        #     ("models/Unigram/Qwen/Qwen2.5-3B-Instruct", devices[2]),
        #     ("models/Unigram/meta-llama/Llama-3.2-3B-Instruct", devices[2]),
        #     ("Qwen/Qwen2.5-3B-Instruct", devices[1]),
        #     ("models/dpo_qwen_3_exponential", devices[2]),
        #     ('meta-llama/Llama-3.1-8B-Instruct', devices[4]),
        #     ("models/dpo_llama_3_all", devices[2])
        # ]
        #
        # for model, device in paraphraser_models:
        #     paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device)
        #     sequences[f'Paraphrase({model})'] = [base_editor, paraphraser]

        model_names = ["models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", 'models/dpo_qwen_3_exponential']
        devices_per_model = len(devices) // len(model_names)
        lambda_weights = [0, 0.2, 0.4, 0.6, 0.8, 1]
        for i, model_name in enumerate(model_names):
            sequences.update(
                RobustnessPipelineFactory._create_weighted_paraphraser_sequences(
                    model_name, devices[i * devices_per_model:(i + 1) * devices_per_model], lambda_weights
                )
            )

        return sequences

    @staticmethod
    def _create_base_paraphrase_sequences(devices: List[str]) -> Dict:
        """Create base set of text editing sequences."""
        paraphraser_models = [
            ("Qwen/Qwen2.5-3B-Instruct", devices[0]),
            ('meta-llama/Llama-3.2-3B-Instruct', devices[0]),
        ]
        sequences = {}
        for model, device in paraphraser_models:
            paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device, num_return_sequences=NUM_PARAPHRASES)
            sequences[f'Paraphrase({model})'] = [paraphraser]
        return sequences

    @staticmethod
    def _create_dipper_sequences(devices: List[str]) -> Dict:
        """Crea sequenze di parafrasi DIPPER su una griglia di budget di diversità (lexical, order)."""
        device = devices[0]
        #DIPPER viene caricato una sola volta in bfloat16 e condiviso da tutte le sequenze
        tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        model = T5ForConditionalGeneration.from_pretrained(
            'kalpeshk2011/dipper-paraphraser-xxl', torch_dtype=torch.bfloat16
        ).to(device)

        #Dizionario delle sequenze di modifica che la pipeline applicherà a ogni testo watermarkato
        sequences = {}
        for lex, order in DIPPER_BUDGETS:
            #Ciclo per il numero di parafrasi previste per ogni budget
            for rep in range(DIPPER_REPEATS):
                #Nome della sequenza
                sequences[f'Dipper_L{lex}_O{order}_r{rep}'] = [
                    DipperParaphraser(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        #Quanto cambiare le parole (0-100)
                        lex_diversity=lex,
                        #Quanto riordinare il contenuto (0-100)
                        order_diversity=order,
                        #Parafrasa a blocchi di 3 frasi
                        sent_interval=3,
                        #Limite di token generati per blocco
                        max_new_tokens=256,
                        #Campionamento della scelta del token più probabile
                        do_sample=True,
                        #Si campiona solo tra i token più probabili che insieme coprono il 75% della probabilità
                        top_p=0.75,
                        #Il filtro sui token è dato solo da top_p
                        top_k=None
                    )
                ]
        return sequences

    @staticmethod
    def _create_partial_dipper_sequences(devices: List[str]) -> Dict:
        """Candidate per i budget piccoli: sinonimi sparsi."""
        device = devices[0]
        sequences = {}
        #Sostituzione di sinonimi sparsa: modifiche distribuite in tutto il testo
        bert_tok = BertTokenizer.from_pretrained('bert-large-uncased')
        bert = BertForMaskedLM.from_pretrained('bert-large-uncased').to(device).eval()
        for ratio in SYNONYM_RATIOS:
            for rep in range(SYNONYM_REPEATS):
                name = f'SynSub_q{round(ratio * 100)}_r{rep}'
                sequences[name] = [SparseSynonymSubstitution(ratio, bert_tok, bert, device=device, seed=rep)]
        #Senza quote di frasi non si carica DIPPER
        if not PARTIAL_FRACTIONS:
            return sequences
        tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        model = T5ForConditionalGeneration.from_pretrained(
            'kalpeshk2011/dipper-paraphraser-xxl', torch_dtype=torch.bfloat16
        ).to(device)
        #Stessi parametri di generazione delle sequenze DIPPER complete
        dipper = DipperParaphraser(tokenizer=tokenizer, model=model, device=device,
                                   lex_diversity=PARTIAL_LEX, order_diversity=0, sent_interval=1,
                                   max_new_tokens=256, do_sample=True, top_p=0.75, top_k=None)
        for fraction in PARTIAL_FRACTIONS:
            for rep in range(PARTIAL_REPEATS):
                name = f'DipperPart_L{PARTIAL_LEX}_p{round(fraction * 100)}_r{rep}'
                sequences[name] = [PartialDipperParaphraser(dipper, fraction=fraction, seed=rep)]
        return sequences

    @staticmethod
    def _create_budget_sequences(model_name: str, devices: List[str]) -> Dict:
        """Crea una sequenza per ogni budget di DIPPER_BUDGETS, tutte con lo stesso parafrasatore."""
        device = devices[0]
        #Il modello viene caricato una volta e condiviso da tutte le sequenze
        loaded = ModelLoader.from_name(model_name, device=device)

        sequences = {}
        for lex, order in QWEN_BUDGETS:
            sequences[f'Budget_L{lex}_O{order}'] = [
                BudgetLLMParaphraser(
                    model=loaded['model'],
                    tokenizer=loaded['tokenizer'],
                    lex_diversity=lex,
                    order_diversity=order,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0
                )
            ]
        return sequences

    @staticmethod
    def _create_llm_sequences(model_name: str, devices: List[str]) -> Dict:
        """Una sola sequenza, senza budget nel prompt."""
        loaded = ModelLoader.from_name(model_name, device=devices[0])
        return {
            'LLM': [BudgetLLMParaphraser(model=loaded['model'], tokenizer=loaded['tokenizer'],
                                         max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0)]
        }

    @staticmethod
    def _create_paraphraser(model_name: str, device: str, **kwargs) -> LLMParaphraser:
        """Create a paraphraser instance."""

        return LLMParaphraser(
            model_config=ModelConfig(model_name=model_name, device=device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            **kwargs
        )

    @staticmethod
    def _create_weighted_paraphraser(model_name: str, device: str, weight: float, **kwargs) -> LLMParaphraser:
        """Create a paraphraser instance."""
        return LLMParaphraser(
            model_config=ModelConfig(model_name=model_name, device=device),
            adapter_ratio=weight,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            **kwargs
        )

    @staticmethod
    def _create_weighted_paraphraser_sequences(model_name: str, devices: List[str], weights: List[int], **kwargs):
        sequences = {}
        for i, weight in enumerate(weights):
            idx = i % len(devices)
            sequences[f'Paraphrase({model_name})_{weight}'] = [
                RobustnessPipelineFactory._create_weighted_paraphraser(
                    model_name, devices[idx], weight, **kwargs
                )
            ]
        return sequences

    @staticmethod
    def _create_ensemble_paraphrasers(model_name: str, devices: List[str], lambda_weights: List[int], **kwargs):
        model_dicts = [ModelLoader.from_name(model_name, device=device) for device in devices]
        base_model_name = model_dicts[0]['model'].config.name_or_path
        base_model_dicts = [ModelLoader.from_name(base_model_name, device=device) for device in devices]
        paraphrasers = {}
        for i, lambda_weight in enumerate(lambda_weights):
            idx = i % len(devices)
            paraphrasers[f'Paraphrase({model_name})_{lambda_weight}'] = [
                RobustnessPipelineFactory._create_ensemble_paraphraser(
                    model_dicts[idx], base_model_dicts[idx], devices[idx], lambda_weight, **kwargs
                )
            ]

        return paraphrasers

    @staticmethod
    def _create_rating_metrics(devices: List[str]) -> Dict:
        """Crea il giudice di qualità usato per valutare testi e parafrasi."""
        #Il giudice va sull'ultima GPU della lista
        judge_device = devices[-1]
        #Carica modello e tokenizer di LLama-3.1-8B-Instruct
        judge = ModelLoader.from_name('meta-llama/Llama-3.1-8B-Instruct', device=judge_device)

        return {
            "llm_judge": LLMTextRater(
                #Spacchetta il dizionario in diverse sezioni
                **judge,
                device=judge_device,
                #Niente chain-of-thought: il giudice risponde senza scrivere prima un'analisi
                cot=False,
                #Di conseguenza bastano pochi token
                max_new_tokens=32,
                #Sceglie sempre il token più probabile
                do_sample=False,
                #Llama ha dei valori di default che farebbero stampare un avviso a ogni chiamata
                top_p=None,
                temperature=None
            )
        }

def write_config(path, data):
    """Scrive un config in modo atomico: più job in parallelo non leggono mai un file scritto a metà."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)

def main():
    import argparse
    from torch.cuda import device_count

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument(
        '--mode',
        type=str,
        default='end_to_end',
        choices=['end_to_end', 'generate_only', 'add_edits', 'base_paraphrase', 'gpt_judge', 'ensemble_paraphrase',
                 're_evaluate', 'dipper_paraphrase', 'dipper_partial_paraphrase', 'budget_paraphrase', 'llm_paraphrase']
    )
    parser.add_argument('--corpus', type=str, default='paper_conclusion', choices=['paper_conclusion', 'markmywords'])
    parser.add_argument('--dataset', type=str, default='eval', choices=['eval', 'train'])
    #Suffisso del file da leggere in modalità 're_evaluate'
    parser.add_argument('--input_suff', type=str, default='_new')
    #Modello che genera i testi watermarkati
    parser.add_argument('--provider', type=str, default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--suff', type=str, required=False, default="_finaleval")
    parser.add_argument('--samples', type=int, default=296)
    #Cartella del parafrasatore addestrato
    parser.add_argument('--paraphraser', type=str, default=None)
    #File JSON con la lista degli id da usare. Se non indicato si usano tutti i testi
    parser.add_argument('--ids_file', type=str, default=None)
    #Se presente, la pipeline gira senza giudice e i voti si calcolano dopo
    parser.add_argument('--no_judge', action='store_true')
    parser.add_argument('--judge_batch_size', type=int, default=16)
    parser.add_argument(
        '--hash_keys',
        type=int,  # Specifies that each input should be converted to an integer
        nargs='+',  # Allows one or more integers (use '*' for zero or more)
        help="A list of hash_keys to use for watermarking",
        default=[15485863]
    )
    args = parser.parse_args()
    directory = f"evaluation/results/{args.provider.replace('/', '_')}"

    # Setup devices
    devices = [f'cuda:{i}' for i in range(device_count())]
    pipeline_factory = RobustnessPipelineFactory(devices)
    # Initialize model and dataset
    provider_config = ModelConfig(
        model_name=args.provider,
        device=devices[0]
    )
    if args.corpus == 'markmywords':
        provider_config.max_new_tokens = 200

    if args.mode in ['generate_only', 'end_to_end']:
        provider_model, provider_tokenizer = ModelLoader(provider_config).load()
    else:
        provider_model, provider_tokenizer = None, _load_tokenizer(provider_config.model_name)
    #Costruisce il dataset dei prompt di generazione
    if args.corpus == 'markmywords':
        dataset = MarkMyWordsDataset(provider_tokenizer, max_samples=args.samples, dataset=args.dataset)
    else:
        dataset = PaperConclusionDataset(provider_tokenizer,
            data_source='dataset/paper_conclusion/arxiv_enhanced.arrow',
            max_samples=args.samples)
    watermark_config_file = f'config/{args.algorithm}.json'
    with open(watermark_config_file, 'r') as f:
        config = json.load(f)

    #Dimensione del vocabolario del provider
    provider_cfg = AutoConfig.from_pretrained(args.provider)
    vocab_size = getattr(provider_cfg, 'text_config', provider_cfg).vocab_size

    #Chiave riservata al test
    TEST_KEY = 15485863
    hash_keys = args.hash_keys
    if not any(k in config for k in ("hash_key", "keys", "key", "mapping_name")):
        hash_keys = [None]  # Just use the default hash key for unsupported algorithms
    #Un run completo della pipeline per ogni chiave
    for hash_key in hash_keys:
        #File di configurazione da passare al watermark
        config_file = watermark_config_file
        #KGW e Unigram: la chiave è un singolo numero
        if "hash_key" in config:
            key_config = dict(config)
            key_config['hash_key'] = hash_key
            config_file = f'config/{args.algorithm}_key{hash_key}.json'
            write_config(config_file, key_config)
        #SynthID: la chiave è una lista di 30 numeri. Se ne genera una nuova
        elif "keys" in config and hash_key != TEST_KEY:
            #Copia la configurazione per non modificare quella originale
            key_config = dict(config)
            #Generatore casuale con seme hash_key che estrae 30 numeri diversi tra 0 e 999
            key_config['keys'] = random.Random(hash_key).sample(range(1000), len(config['keys']))
            #Salva la configurazione in un file separato
            config_file = f'config/{args.algorithm}_key{hash_key}.json'
            write_config(config_file, key_config)
        #EXPEdit: la chiave è un singolo numero, salvato in un config separato per ogni chiave
        elif "key" in config:
            key_config = dict(config)
            key_config['key'] = hash_key
            config_file = f'config/{args.algorithm}_key{hash_key}.json'
            write_config(config_file, key_config)
        #SIR: la chiave è la mappatura dei token, ognuna diversa e riproducibile
        elif "mapping_name" in config:
            key_config = dict(config)
            mapping_file = f'watermark/sir/mapping/{config["scale_dimension"]}_mapping{vocab_size}_key{hash_key}.json'
            if not os.path.exists(mapping_file):
                rng = random.Random(hash_key)
                with open(mapping_file, 'w') as f:
                    json.dump([rng.randrange(config['scale_dimension']) for _ in range(vocab_size)], f)
                key_config['mapping_name'] = mapping_file
                config_file = f'config/{args.algorithm}_key{hash_key}.json'
                write_config(config_file, key_config)

        watermark_args = {
            'algorithm_name': args.algorithm,
            'algorithm_config': config_file,
            'transformers_config': TransformersConfig(
                model=provider_model,
                tokenizer=provider_tokenizer,
                vocab_size=vocab_size,
                device=provider_config.device,
                max_new_tokens=provider_config.max_new_tokens,
                do_sample=provider_config.do_sample
            )
        }
        if args.suff:
            suff = args.suff
        else:
            suff = "_gen" if args.mode == 'generate_only' else "_edit"
        # Create and run pipeline based on mode
        pipeline = pipeline_factory.create_pipeline(
            dataset=dataset,
            mode=args.mode,
            filename=f'{directory}/{args.algorithm}{suff}{hash_key}.tsv',
            paraphraser=args.paraphraser,
            no_judge=args.no_judge,
            judge_batch_size=args.judge_batch_size
        )
        if args.mode in ['generate_only', 'end_to_end']:
            pipeline.evaluate(watermark_args)
        elif args.mode in ['gpt_judge', 're_evaluate']:
            pipeline.add_quality(f'{directory}/{args.algorithm}{args.input_suff}{hash_key}.tsv')
        else:
            if args.mode == 'base_paraphrase':
                pipeline.edit_expansion = len(pipeline.config.edit_sequences) * NUM_PARAPHRASES
            only_ids = set(json.load(open(args.ids_file))) if args.ids_file else None
            pipeline.add_edits(f'{directory}/{args.algorithm}_finaleval{hash_key}.tsv', watermark_args, only_ids=only_ids)

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()