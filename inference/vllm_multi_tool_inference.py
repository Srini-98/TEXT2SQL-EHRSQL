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
import ast
import json

torch.cuda.manual_seed(42)
torch.manual_seed(42)
table_lis = []

with open("./prompts/main_multi_tool_prompt_input_prof.txt", "r") as f:
    prompt_input = f.read()

with open("./prompts/main_multi_tool_prompt_output_prof.txt", "r") as g:
    prompt_output = g.read()



def get_foreign_keys(keys_lis , tup_lis , foreign_keys):
    st = []
    for table_pair in tup_lis:
        if table_pair[0].lower().strip() in keys_lis and table_pair[1].lower().strip() in keys_lis: 
            st.append(foreign_keys[tup_lis.index(table_pair)])
    return st
    

def format_input(question , prompt , table_format):
    return prompt.format(tables_list = table_format , question = question)

def format_input_code(question , prompt , foreign_keys ,  table_format , fn_call ,  table_json , tup_lis):
    st = prompt.format(tables_list=table_format , question = question)
    st = st + fn_call + "\n\n" + "Table schema:\n"
    lis = fn_call.split("get_schema(")[1].split(")")[0]
    if "get_keys" in fn_call:
        keys_lis = fn_call.split("get_keys(")[1].split(")")[0] 
        keys_lis = ast.literal_eval(keys_lis)
        if len(keys_lis) == 1:
            foreign_keys = "Not Required"
        else:
            foreign_keys = get_foreign_keys(keys_lis , tup_lis , foreign_keys)
            foreign_keys = " , ".join(foreign_keys)
            if foreign_keys == "":
                foreign_keys = "Not Required"
    else:
        foreign_keys = "Not Required"
    actual_list = ast.literal_eval(lis)
    table_st = ""
    
    for j in actual_list:
        try:
            table_json_value = table_json[j]
            table_st = table_st +  table_json[j] + "\n\n"
        except:
            print("Illegal table" , j)
            continue

    st = st + table_st + "\n" + "Foreign Keys:\n" + foreign_keys + "\n\n" 

    return st

    
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
    foreign_st = None ,
    inference_stage = None , 
    table_func_file = None , 
    pred_table_path = None , 
    batch_size=6
):
    print("dataset used for eval" , data_used)
    print("output_file" , output_path)
    print("valid original" , valid_original)
    print("table st" , table_st)
    print("foreign st" , foreign_st)
    print("inference stage" , inference_stage)
    print("table func file" , table_func_file)

    with open(f"./prompts/{table_st}" , "r") as g:
        table_format = g.read().split("\n\n")

    with open(f"./prompts/{foreign_st}" , "r") as h:
        foreign_keys = h.read().split(",\n")

    with open(f"./table_data/{table_func_file}" , "r") as f:
        table_json = json.load(f)
    

    table_lis = []
    for i in table_format:
        t = i.split("(")[0].split("TABLE ")[1]
        table_lis.append(t.lower())
    
    f_lis = []
    tup_lis = []
    for i in foreign_keys:
        f_lis.append(i.lower())
        t_1 = i.split("=")[0].split(".")[0]
        t_2 = i.split("=")[1].split(".")[0]
        tup_lis.append([t_1,t_2])


    if inference_stage == 2:
        with open(pred_table_path , "r") as t:
            predicted_tables = t.read().split("\n")
    
    with open(valid_original , "r") as g:
        data_original = json.load(g)
    
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
    
    out_eval = {}
    start_ind = 0
    
    if inference_stage == 1:
        sampling_param = SamplingParams(max_tokens=600 , temperature=0 ,logprobs=32000 , stop = ["</function_call>"]) 
    else: 
        sampling_param = SamplingParams(max_tokens=600 , temperature=0 ,logprobs=32000)
    
    start_time = time.time()
    j = 0
    flag = 0
    if inference_stage == 1:
        output_file = open(f"{output_path}" , "w")

    out_eval = {}
    num_beams=1
    start_ind = 0
    print("batch size is" , batch_size)
    for i in range(0 , len(data['train']) , batch_size):
        user_prompt = []
        for k in range(0 , batch_size):
            if inference_stage == 1:
                element = format_input(data['train'][i+k]['question'] , prompt_input , table_lis) if i + k < len(data['train']) else None
            else:
                element = format_input_code(question=data['train'][i+k]['question'] , foreign_keys=foreign_keys ,  prompt=prompt_input , table_format=table_lis , fn_call= predicted_tables[i+k], table_json=table_json , tup_lis = tup_lis) if i + k < len(data['train']) else None

            if element is not None and element != "ERROR":
                user_prompt.append(element)
            else:
                print("none element")
                print(element)
        outputs = model.generate(user_prompt, sampling_params=sampling_param)

        for ct , output in enumerate(outputs):
            output_lis = []
            for o in output.outputs:
                final_output = o.text
                output_lis.append(final_output)

                if inference_stage == 1:
                    print("final output" , final_output)
                    output_file.write(final_output + "</function_call>" + "\n")
                else:
                    try:
                        code = final_output.split("SQLCODE:\n")[1]
                    except:
                        print("final output due to ERROR" , element + final_output)
                        code = "ERROR"

                    print("final output" , code)
                    
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
                    result['pred'] = code
                    entropy = sequences_entropy[0].tolist()
                    result['sequence_entropy'] = tuple(entropy)
                    result['db_id'] = db_id[start_ind]
                    result['is_impossible'] =  impossible_lis[start_ind]
                    out_eval[id[start_ind]] = result
                    start_ind = start_ind + 1

            if i == 0 and flag == 0:
                flag = 1
                print("output is")
                print(user_prompt[0] + final_output)
            print("-------------------------------------")

    if inference_stage == 2:
        with open(output_path , "w" ) as f:
            json.dump(out_eval , f)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if inference_stage == 1:
        output_file.close()
    print(f"Execution time: {elapsed_time:.2f} seconds")

    
def run_script(
    model_name: str,
    peft_model=None,
    tp_size=1,
    max_new_tokens=100,
    batch_size=6,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8,
    dataset_path = None,
    output_path = None,
    data_used = None , 
    valid_original = None , 
    table_st = None , 
    foreign_st = None , 
    inference_stage = None ,
    table_func_file = None , 
    pred_table_path = None , 
):
    print("starting to load model")
    print('tp_size' , tp_size)
    print("batch size" , batch_size)
    model = load_model(model_name, tp_size)
    print("Finished loading model")

    print(len(model.get_tokenizer()))
    main(model = model, max_new_tokens= int(max_new_tokens), user_prompt= user_prompt, top_p= int(top_p), temperature= int(temperature) , 
         dataset_path=dataset_path , output_path = output_path , data_used = data_used , valid_original = valid_original , table_st = table_st , foreign_st = foreign_st , inference_stage = inference_stage ,
        table_func_file = table_func_file , pred_table_path = pred_table_path , batch_size= batch_size)

if __name__ == "__main__":
    fire.Fire(run_script)