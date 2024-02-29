# TEXT2SQL-EHRSQL
This work builds on top of the datasets released with https://github.com/glee4810/EHRSQL. Use the instructions mentioned in the link to download the database and the datasets. 

## Set up Environment

Set up conda environment using (python>=3.10):

```
pip install -r requirements.txt
```

## Training
Two models are used in this work - Mistral and Llama 2. Download the models from the link below:
Mistral7B : https://huggingface.co/mistralai/Mistral-7B-v0.1
Llama27B  : https://huggingface.co/meta-llama/Llama-2-7b

Both the bash scripts start a FSDP process. Set the --nnodes(number of nodes) and  --nproc-per-node(number of GPUs) in [run.sh](https://github.com/Srini-98/TEXT2SQL-EHRSQL/blob/master/run.sh) and [run_multi.sh](https://github.com/Srini-98/TEXT2SQL-EHRSQL/blob/master/run_multi.sh) as per the compute available. 
The training to reproduce the results can be performed using the following command:

### No Predicting tables or keys 
Note: Pass the path of the model, the learning rate, the dataset path, output directory, model name and dataset name as parameters to the script.  The training set for both the datasets are assumed to be named 'train_mimic.json' , 'train_eicu.json'. Change the name in the bash command according to the file name.

##### Model Mistral, Dataset Mimic:
```
bash run.sh https://huggingface.co/mistralai/Mistral-7B-v0.1 1e-5 train_mimic.json ./mistral_main_prompt_mimic/ mistral mimic
```

##### Model Mistral, Dataset EICU:
```
bash run.sh https://huggingface.co/mistralai/Mistral-7B-v0.1 1e-5 train_eicu.json ./mistral_main_prompt_eicu/ mistral eicu
```

##### Model Llama2, Dataset Mimic:
```
bash run.sh https://huggingface.co/meta-llama/Llama-2-7b 1e-5 train_mimic.json ./llama2_main_prompt_mimic/ llama2 mimic
```

##### Model Llama2, Dataset EICU:
```
bash run.sh https://huggingface.co/meta-llama/Llama-2-7b 1e-5 train_eicu.json ./llama2_main_prompt_eicu/ llama2 eicu
```

### Predicting tables and keys 

The same steps from the previous section have to be followed for multi funciton training. 

##### Model Mistral, Dataset Mimic:

```
bash run_multi.sh https://huggingface.co/mistralai/Mistral-7B-v0.1 1e-5 train_multistep.json ./mistral_multitool_prompt_mimic/ mistral mimic
```

## Inference
Go to the inference folder for instructions to perform inference on the development set.

## Evaluation 
For evalauting the code generated by the finetuned models we follow the same method that is used in  [EHRSQL](https://github.com/glee4810/EHRSQL?tab=readme-ov-file#evaluation). Refer to the link and follow the same procedure for evaluation, or follow the steps below.
Install the `func-timeout` if you have not.
```
pip install func-timeout
```

Run the following. The generated file will be in the same directory as the json input gile
```
./eval/run_abstain_entropy.sh {json file from the inference step} {formatted output filename}
```

Prepare the database in the `database` directory and run the following to get the score. The {score output file} variable is not used, the score will be printed on terminal.
```
./eval/run_ehrsql_eval.sh {original validation json} {formatted output filename} {score output file} {dataset name}
```
 
