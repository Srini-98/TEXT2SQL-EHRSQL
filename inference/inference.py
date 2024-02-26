import fire
import json
import torch
from vllm import LLM
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import time
from tqdm import tqdm
from datasets import load_from_disk

torch.cuda.manual_seed(42)
torch.manual_seed(42)


def format_input(st , prompt , table_format , foreign_keys):
    prompt = prompt.format(user_question=st['question'],
                  table_metadata_string=table_format,
                  foreign_keys = " , ".join(foreign_keys)
                  )
    return prompt



def load_model(model_name, tp_size=1):
    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm

def main(
    model,
    max_new_tokens=300,
    user_prompt=None,
    top_p=0,
    dataset_path = None,
    temperature=0,
    output_path = None,
    data_used = None , 
    valid_original = None , 
    table_st = None , 
    foreign_st = None
):
    
    print("dataset used for eval" , data_used)
    print("output_file" , output_path)
    print("valid original" , valid_original)
    print("table st" , table_st)
    print("foreign st" , foreign_st)
    with open(valid_original , "r") as g:
        data_original = json.load(g)
    
    with open("./prompts/main_prompt.txt", "r") as f:
        prompt = f.read()

    with open(f"./prompts/{table_st}" , "r") as g:
        table_format = g.read()

    with open(f"./prompts/{foreign_st}" , "r") as g:
        foreign_keys = g.read().split(",\n")

    label_lis = []
    impossible_lis = []
    db_id = []
    id = []
    question_lis = []
    for i in data_original:
        question_lis.append(i['question'])
        label_lis.append(i['query'])
        impossible_lis.append(i["is_impossible"])
        db_id.append(i["db_id"])
        id.append(i["id"]) 

    
    print("number of questions" , len(question_lis))
    data = load_dataset("json", data_files= dataset_path)
    print(data)
    out_eval = {}
    start_ind = 0
    num_beams = 3
    sampling_param = SamplingParams(max_tokens=600 , temperature=0 ,logprobs=32000) #use_beam_search=True , best_of=num_beams)
    start_time = time.time()
    j = 0
    batch_size = 6


    for i in range(0, len(data['train']), batch_size):
        user_prompt  = []
        for k in range(0 , batch_size):
            element = format_input(data['train'][i+k] , prompt , table_format , foreign_keys) if i + k < len(data['train']) else None  #,  table_format , foreign_keys) if i + k < len(data['train']) else None
            if element is not None:
                user_prompt.append(element)
        
        if i  == 0:
            print(user_prompt[0])
        
        outputs = model.generate(user_prompt, sampling_params=sampling_param)

        for output in outputs:
            output_lis = []
            for o in output.outputs:
                final_output = o.text
                print("final output" , final_output)
                output_lis.append(final_output)
            print("-------------------------------------")
            log_probs = output.outputs[0].logprobs
            final_scores = []
            t = () 
            for log_token in log_probs:
                lis = []
                scores = []
                for tok , val in log_token.items():
                    lis.append(val)
                scores.append(lis)
                t = t + (torch.tensor(scores),)

            logits = torch.stack(t, dim=1)[::int(num_beams/1)]
            logits = logits.cpu()
            output_prob = torch.softmax(logits, dim=2).float()
            log_prob = torch.log_softmax(logits, dim=2).float()
            output_prob = torch.softmax(logits, dim=2).float()
            log_prob = torch.log_softmax(logits, dim=2).float()
            sequences_entropy = (torch.sum(output_prob * log_prob, dim=2) * (-1) ).numpy()
            result = {}
            
            result['question'] = question_lis[start_ind]
            result['real'] = label_lis[start_ind]
            result['pred'] = output_lis[0]
            entropy = sequences_entropy[0].tolist()
            result['sequence_entropy'] = tuple(entropy)
            result['db_id'] = db_id[start_ind]
            result['is_impossible'] =  impossible_lis[start_ind]
            out_eval[id[start_ind]] = result
            start_ind = start_ind + 1
        

    with open(output_path , "w" ) as f:
        json.dump(out_eval , f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

def run_script(
    model_name: str,
    peft_model=None,
    tp_size=1,
    max_new_tokens=100,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8,
    dataset_path = None,
    output_path = None,
    data_used = None , 
    valid_original = None , 
    table_st = None , 
    foreign_st = None
):
    print("starting to load model")
    model = load_model(model_name, tp_size)
    print("Finished loading model")

    print(len(model.get_tokenizer()))
    main(model = model, max_new_tokens= int(max_new_tokens), user_prompt= user_prompt, top_p= int(top_p), temperature= int(temperature) , 
         dataset_path=dataset_path , output_path = output_path , data_used = data_used , valid_original = valid_original , table_st = table_st , foreign_st = foreign_st)

if __name__ == "__main__":
    fire.Fire(run_script)