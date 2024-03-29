## Inference 
The inference and evaluation is performed on the official development set released in https://github.com/glee4810/EHRSQL.
Download the datasets and use the [processing](process.py) script to get a processed version of the development set. Perform this processing for both the datasets. 

Name the files: 
1. 'valid_mimic.json' (for the downloaded original file)
2. 'processed_valid_mimic.json' (for the processed file)

1. 'valid_eicu.json' (for the downloaded original file)
2. 'processed_valid_eicu.json' (for the processed file)


### One Step Inference ( No tables or keys are predicted):
Run the following command for starting the inference process: 

##### Mimic3
```
bash run_vllm_basic.sh mimic3 mimic3 processed_valid_mimic.json valid_mimic.json model_path
```

##### EICU
```
bash run_vllm_basic.sh eicu eicu processed_valid_eicu.json valid_eicu.json model_path
```

Note: model_path here refers to the path of the finetuned checkpoint from the training process. 


### Two Step Inference ( tables and keys are predicted before generating the code):
Run the following command for starting the inference process: 

##### Mimic3
```
bash run_vllm_2_step.sh mimic3 mimic3 processed_valid_mimic.json valid_mimic.json model_path
```

##### EICU
```
bash run_vllm_2_step.sh eicu eicu processed_valid_eicu.json valid_eicu.json  model_path
```
