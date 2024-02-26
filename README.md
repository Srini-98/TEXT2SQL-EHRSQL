# TEXT2SQL-EHRSQL
This work builds on top of the datasets released with https://github.com/glee4810/EHRSQL. Use the instructions mentioned in the link to download the database and the datasets. 

## Training
Two models are used in this work - Mistral and Llama 2. Download the models from the link below:
Mistral7B : https://huggingface.co/mistralai/Mistral-7B-v0.1
Llama27B  : https://huggingface.co/meta-llama/Llama-2-7b

The training to reproduce the results can be performed using the following command:

#### No Predicting tables or keys
```
bash run.sh https://huggingface.co/mistralai/Mistral-7B-v0.1 1e-5 train.json ./mistral_main_prompt/ mistral
```
Note: Pass the path to the model , the learning rate , the dataset path , output directory and model name as parameters to the script. 

#### Predicting tables and keys 
```
bash run_multi.sh https://huggingface.co/mistralai/Mistral-7B-v0.1 1e-5 train_multistep.json ./mistral_multitool_prompt/ mistral
```
Note: Pass the path to the model , the learning rate , the dataset path , output directory and model name as parameters to the script. 

## Inference
