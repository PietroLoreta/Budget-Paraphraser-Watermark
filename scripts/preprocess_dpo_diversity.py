import sys
import json
import random
from collections import defaultdict
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

prompt = (
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences.\n"
    "Ensure that the final output contains the same information as the original text and has roughly the same length.\n"
    "Do not leave out any important details when rewriting in your own voice. Do not include any information that is not "
    "present in the original text. Do not respond with a greeting or any other extraneous information. "
    "Skip the preamble. Just rewrite the text directly."
)

