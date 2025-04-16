import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback

import torch
import transformers
import sklearn
import scipy
import numpy as np
import re
from torch.utils.data import Dataset

import sys

# current_path = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_path)
sys.path.append('../')


from model.rnalm.modeling_rnalm import RnaLmForNucleotideLevel
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForNucleotideLevel
from model.rnabert.modeling_rnabert import RnaBertForNucleotideLevel
from model.rnamsm.modeling_rnamsm import RnaMsmForNucleotideLevel
from model.splicebert.modeling_splicebert import SpliceBertForNucleotideLevel
from model.utrbert.modeling_utrbert import UtrBertForNucleotideLevel
from model.utrlm.modeling_utrlm import UtrLmForNucleotideLevel
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
# Model Arguments
MODEL_PATH = os.path.abspath("../checkpoint/opensource/splicebert-ms1024/")
model_max_length = 1026
token_type = "single"
model_type = "splicebert-ms1024"

# Data Arguments
data_path = "../data/SpliceAI"
data = ""  # Empty string as shown in the script
data_train_path = "train.csv"
data_val_path = "val.csv"
data_test_path = "test.csv"

# Training Arguments
run_name = f"{model_type}_{data}"
output_dir = f"./outputs/ft/rna-all/SpliceAI/splicebert-ms1024/{data}"
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 3e-5
num_train_epochs = 30
fp16 = True
save_steps = 400
evaluation_strategy = "steps"
eval_steps = 200
warmup_steps = 50
logging_steps = 200
overwrite_output_dir = True
log_level = "info"
seed = 666
cache_dir = "./cache"

# If you need to create the actual HuggingFace arguments objects:
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: str = MODEL_PATH
    model_max_length: int = model_max_length
    token_type: str = token_type
    model_type: str = model_type

@dataclass
class DataArguments:
    data_path: str = data_path
    data_train_path: str = data_train_path
    data_val_path: str = data_val_path
    data_test_path: str = data_test_path

@dataclass
class TrainingArguments:
    run_name: str = run_name
    output_dir: str = output_dir
    per_device_train_batch_size: int = per_device_train_batch_size
    per_device_eval_batch_size: int = per_device_eval_batch_size
    gradient_accumulation_steps: int = gradient_accumulation_steps
    learning_rate: float = learning_rate
    num_train_epochs: int = num_train_epochs
    fp16: bool = fp16
    save_steps: int = save_steps
    evaluation_strategy: str = evaluation_strategy
    eval_steps: int = eval_steps
    warmup_steps: int = warmup_steps
    logging_steps: int = logging_steps
    overwrite_output_dir: bool = overwrite_output_dir
    log_level: str = log_level
    seed: int = seed
    model_type: str = model_type
    model_max_length: int = model_max_length
    token_type: str = token_type
    cache_dir: str = cache_dir


from downstream.train_spliceai import SupervisedDataset, DataCollatorForSupervisedDataset

# Create instances of the argument classes
model_args = ModelArguments()
data_args = DataArguments()
training_args = TrainingArguments()

tokenizer = OpenRnaLMTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)

# train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
#                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), )
#                                     # kmer=data_args.kmer)
# val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
#                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), )
#                                     # kmer=data_args.kmer)
test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                    data_path=os.path.join(data_args.data_path, data_args.data_test_path), )
                                    # kmer=data_args.kmer)
# data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
# print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')


print(training_args.model_type)
print(f'Loading {training_args.model_type} model')
model = SpliceBertForNucleotideLevel.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    num_labels=test_dataset.num_labels,
    trust_remote_code=True,
    problem_type="single_label_classification",
    token_type=training_args.token_type,
    tokenizer=tokenizer,
)   

sequences = ["AUUCCGAUUCCGAUUCCG"]
output = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="longest", max_length = 1026, truncation=True)
input_ids = output["input_ids"]
attention_mask = output["attention_mask"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.half().to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

batch_size = input_ids.shape[0]
sequence_length = input_ids.shape[1]

weight_mask = torch.ones_like(attention_mask).half().cuda()

post_token_length = torch.ones_like(attention_mask).half().cuda()

embedding = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    weight_mask=weight_mask,
    post_token_length=post_token_length
)

print(embedding.shape)