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
import ast
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"
table_lis = []



def format_input(question , prompt , table_format):
    return prompt.format(tables_list = table_format , question = question)

def format_output(function_lis , schema , code , get_keys ,  prompt_output , foreign_keys):
    key_fn = ""
    if get_keys == "[]":
        key_fn = ""
    else:
        key_fn = f" get_keys({function_lis})"
    
    print("get keys: " , key_fn)
    
    if foreign_keys == "[]":
        foreign_keys = "Not Required"
    else:
        foreign_keys = " , ".join(ast.literal_eval(foreign_keys))
    return prompt_output.format(function_lis = function_lis , get_keys = key_fn , schema = schema , code = code , foreign_keys = foreign_keys)


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

def preprocess(train_on_inputs: bool , samples: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer , prompt_input , table_lis , prompt_output) -> Dict:
    sources = [f"{format_input(question=st , prompt=prompt_input , table_format=table_lis)}" for st in samples['question']]
    targets = [f"{format_output(function_lis=function_lis , schema=schema , code=code , prompt_output=prompt_output , get_keys = get_keys , foreign_keys = f_keys)}{tokenizer.eos_token}" for function_lis , schema , code , get_keys , f_keys in zip( samples['table_lis'] , samples['table_query'],samples['query'] , samples['foreign_keys_val'] , samples['foreign_keys_val'])]
    examples = [s + t for s , t in zip(sources , targets)]
    examples_tokenized , source_tokenized = [ _tokenize_fn(strings , tokenizer) for strings in (examples , sources)]
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label , source_len in zip(labels , source_tokenized['input_id_lens']):
        if not train_on_inputs:
            label[:source_len] = IGNORE_INDEX

    final_l = []
    for val in labels:
        res = []
        i = 0
        s = 0
        while i < len(val):
            if val[i] == 13 and val[i+1] == 13:
                break
            res.append(val[i].item())
            i = i + 1

        res.append(-100)
        res.append(-100)
        i = i + 2
        while i < len(val):
            if val[i] == 13 and val[i+1] == 13 and val[i+2] == 13:
                break
            res.append(-100)
            i = i + 1

        res.append(-100)
        res.append(-100)
        res.append(-100)
        i = i + 3


        while i < len(val):
            if val[i] == 13 and val[i+1] == 13:
                break
            res.append(-100)
            i = i + 1
        
        res.append(-100)
        res.append(-100)
        i = i + 2
        
        while i < len(val):
            res.append(val[i].item())
            i = i + 1
        
        final_l.append(torch.tensor(res))

    return dict(input_ids = input_ids , labels = final_l)

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

def filter_long_samples(samples: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer , prompt_input , table_lis , prompt_output) -> Dict:
    sources = [f"{format_input(question=st , prompt=prompt_input , table_format=table_lis)}" for st in samples['question']]
    targets = [f"{format_output(function_lis=function_lis , schema=schema , code=code , prompt_output=prompt_output , get_keys = get_keys , foreign_keys = f_keys)}{tokenizer.eos_token}" for function_lis , schema , code , get_keys , f_keys in zip( samples['table_lis'] , samples['table_query'],samples['query'] , samples['foreign_keys_val'] , samples['foreign_keys_val'])]
    examples = [s + t for s , t in zip(sources , targets)]
    return _filter_tokenize_fn(examples , tokenizer)


class SuperVisedDataset(Dataset):

    def __init__(self , train_on_inputs: bool , tokenizer: transformers.PreTrainedTokenizer , dataset , dataset_name):

        super(SuperVisedDataset , self).__init__()
        workers = math.ceil(os.cpu_count() / dist.get_world_size())
        logging.warning(f"Tokenizing with {workers} workers")

        if dataset_name == "mimic":
            table_name = "tables.txt"
        elif dataset_name == "eicu":
            table_name = "tables_eicu.txt"
        
        with open(f"./prompts/{table_name}" , "r") as g:
            table_format = g.read().split("\n\n")

        with open("./prompts/main_table_pred_prompt_input_prof.txt", "r") as f:
            prompt_input = f.read()

        with open("./prompts/main_table_pred_prompt_output_prof.txt", "r") as g:
            prompt_output = g.read()

        for i in table_format:
            t = i.split("(")[0].split("TABLE ")[1]
            table_lis.append(t.lower())

        dataset = dataset.filter(
            lambda x: filter_long_samples(x , tokenizer , prompt_input , table_lis , prompt_output) ,
            batched=True,
            batch_size=1000,
            num_proc=workers
        ).map(
            lambda x: preprocess(train_on_inputs , x , tokenizer , prompt_input , table_lis , prompt_output),
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