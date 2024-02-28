from dataclasses import dataclass
import os 
from typing import List, Optional, Tuple, Union , Dict , Sequence
import datasets 
import logging 
import torch.distributed as dist
import torch 
import transformers
import copy 
import math
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"



def format_input(st , prompt , table_format , foreign_keys):
    prompt = prompt.format(user_question=st,
                  table_metadata_string=table_format,
                  foreign_keys = " , ".join(foreign_keys)
                  )
    return prompt

def format_output(st):
    return f"""{st}"""

def _tokenize_fn(strings: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer) ->Dict:
    tokenized_list = [tokenizer(text , return_tensors="pt" , padding=False , max_length=tokenizer.model_max_length , truncation=True) for text in strings]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_id_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids = input_ids,
        labels = labels,
        input_id_lens = input_id_lens,
        labels_lens = labels_lens,
    )

def preprocess(train_on_inputs: bool , samples: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer , prompt , table_format , foreign_keys) -> Dict:
    sources = [f"{format_input(st , prompt , table_format , foreign_keys)}" for st in samples['question']]
    targets = [f"{format_output(st)}{tokenizer.eos_token}" for st in samples['query']]
    examples = [s + t for s , t in zip(sources , targets)]
    examples_tokenized , source_tokenized = [ _tokenize_fn(strings , tokenizer) for strings in (examples , sources)]
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label , source_len in zip(labels , source_tokenized['input_id_lens']):
        if not train_on_inputs:
            label[:source_len] = IGNORE_INDEX

    return dict(input_ids = input_ids , labels = labels)

def _filter_tokenize_fn(strings: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    samples = []
    for text in strings:
        tokens = tokenizer(text , return_tensors="pt" , padding=False , 
                           max_length=tokenizer.model_max_length , truncation=True)

        if tokens.input_ids.squeeze().numel() < tokenizer.model_max_length:
            samples.append(True)
        else:
            samples.append(False)
    return samples

def filter_long_samples(samples: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer , prompt , table_format , foreign_keys) -> Dict:
    sources = [f"{format_input(st , prompt , table_format , foreign_keys)}" for st in samples['question']]
    targets = [f"{format_output(st)}{tokenizer.eos_token}" for st in samples['query']]
    examples = [s + t for s , t in zip(sources , targets)]
    return _filter_tokenize_fn(examples , tokenizer)


class SuperVisedDataset(Dataset):

    def __init__(self , train_on_inputs: bool , tokenizer: transformers.PreTrainedTokenizer , dataset , dataset_name):
        super(SuperVisedDataset , self).__init__()
        workers = math.ceil(os.cpu_count() / dist.get_world_size())
        logging.warning(f"Tokenizing with {workers} workers")
        
        if dataset_name == "mimic":
            table_name = "tables.txt"
            foreign_keys = "foreign_keys.txt"
        elif dataset_name == "eicu":
            table_name = "tables_eicu.txt"
            foreign_keys = "foreign_keys_eicu.txt"

        print("table name" , table_name)
        print("foreign keys" , foreign_keys)
        
        with open("./prompts/main_prompt_prof.txt", "r") as f:
            prompt = f.read()

        with open(f"./prompts/{table_name}" , "r") as g:
            table_format = g.read()

        with open(f"./prompts/{foreign_keys}" , "r") as h:
            foreign_keys = h.read().split(",\n")

        dataset = dataset.filter(
            lambda x: filter_long_samples(x , tokenizer , prompt , table_format , foreign_keys) ,
            batched=True,
            batch_size=1000,
            num_proc=workers
        ).map(
            lambda x: preprocess(train_on_inputs , x , tokenizer , prompt , table_format , foreign_keys),
            batched=True,
            batch_size=1000,
            num_proc=workers,
        )
        
        self.input_ids = dataset['input_ids']
        self.labels = dataset['labels']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self , idx) -> Dict[str , torch.Tensor]:
        return dict(
            input_ids = torch.tensor(self.input_ids[idx]),
            labels = torch.tensor(self.labels[idx]),
        )

@dataclass
class DataCollatorForSuperVisedDataset(object):
    """collate examples for supervised finetuning"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self , instances: Sequence[Dict]) -> Dict[str , torch.Tensor]:
        input_ids , labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids" , "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids , batch_first=True , padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels , batch_first=True , padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        )