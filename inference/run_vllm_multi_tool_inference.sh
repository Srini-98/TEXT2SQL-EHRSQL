# model_path=/home/project/11003644/srini/full_finetuning/outputs/data/projects/11003644/srini/models/Mistral-7B-v0.1/d416d122-9f69-49c5-8db6-bf103d8f12f6/epoch_2_step_524/
peft_path=" "
top_p=1
temperature=0

device=0
output_sub_path=/home/project/11003644/srini/full_finetuning/mimic3_inference_outputs/

echo "starting inference"
data_used="mimic3"
sub_data_mimic="eicu"
count=1


table_st=""  # Variable to store the result
foreign_st=""

if [ "$sub_data_mimic" = "eicu" ]; then
    table_st=tables_eicu.txt
    foreign_st=foreign_keys_eicu.txt
    table_func_file=tables_func_eicu.json
    dataset_path=/home/project/11003644/srini/full_finetuning/mimic_data/EICU_main_valid.json
    valid_original=/home/project/11003644/srini/full_finetuning/mimic_data/EICU_valid.json
else
  table_st=tables.txt
  foreign_st=foreign_keys.txt
  table_func_file=tables_func_mimic.json
  dataset_path=/home/project/11003644/srini/mimic3/final_data/valid_data.json
  valid_original=/home/project/11003644/srini/mimic3/final_data/valid_original.json
fi

echo dataset path $dataset_path
echo valid path $valid_original
echo table $table_st
echo foreign key $foreign_st
inference_stage=1

lr=1.5e-5

for (( i=1; i<3; i++ ))
do
  echo "Count: $i"
  inference_stage=$i
  echo "inference stage $inference_stage"
  
  if [ "$inference_stage" = 1 ]; then
      output_path=./multi_tool_eicu_llama_2_epoch1_training_2e-5.txt
      pred_table_path=''
  else
    pred_table_path=./multi_tool_eicu_llama_2_epoch1_training_2e-5.txt
    output_path=./multi_tool_eicu_llama_2_epoch1_training_2e-5.json
  fi

  model_path=/home/project/11003644/srini/full_finetuning/llama2_eicu_multitool_2e-5_1_epoch/epoch_1_step_551
  #/home/project/11003644/srini/full_finetuning/llama2_mimic3_weights_multitool/epoch_3_step_554/
  #/home/project/11003644/srini/full_finetuning/llama2_eicu_function_call_weights_lr_0.5e-5_test_size_0.05/epoch_3_step_551
  #/home/project/11003644/srini/full_finetuning/llama2_mimic3_function_call_weights/epoch_3_step_525
  #/home/project/11003644/srini/full_finetuning/multitool_pred_weights_mimic3_0.5e-5_test_size0.05/epoch_3_step_554
  echo "output file" $output_path
  echo "predicted table path" $pred_table_path
  echo "model path" $model_path

  
  CUDA_VISIBLE_DEVICES=$device nohup python ./vllm_multi_tool_inference.py \
      --model_name $model_path \
      --top_p $top_p \
      --temperature $temperature \
      --dataset_path $dataset_path \
      --data_used  $data_used \
      --output_path $output_path \
      --valid_original $valid_original \
      --foreign_st $foreign_st \
      --table_st $table_st \
      --tp_size 1 \
      --batch_size 6 \
      --inference_stage $inference_stage \
      --table_func_file $table_func_file \
      --pred_table_path $pred_table_path > vllm_table_pred_inference_logs_llama2_mimic_epoch1.txt

done
