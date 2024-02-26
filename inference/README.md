## Inference 
The inference and evaluation is performed on the official development set released in https://github.com/glee4810/EHRSQL.
Download the datasets and use the [processing](process.py) script to get a processed version of the developmnet set. Perform this processing for both the datasets. 


### One Step Inference ( No tables or keys are predicted):
Run the following command for starting the inference process: 

##### Mimic3
```
bash run_vllm_basic.sh mimic3 mimic3 valid_mimic.json processed_valid_mimic.json model_path
```

##### EICU
```
bash run_vllm_basic.sh eicu eicu valid_eicu.json processed_valid_eicu.json model_path
```

Note: model_path here refers to the path of the finetuned checkpoint from the training process. 


### Two Step Inference ( tables and keys are predicted before generating the code):
Run the following command for starting the inference process: 

##### Mimic3
```
bash run_vllm_2_step.sh mimic3 mimic3 valid_mimic.json processed_valid_mimic.json model_path
```

##### EICU
```
bash run_vllm_2_step.sh eicu eicu valid_eicu.json processed_valid_eicu.json model_path
```
