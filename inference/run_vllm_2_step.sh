top_p=1
temperature=0

device=0

echo "starting inference"
data_used=$1
sub_data_mimic=$2
model_path=$3
count=1

table_st=""  
foreign_st=""

if [ "$sub_data_mimic" = "eicu" ]; then
    table_st=tables_eicu.txt
    foreign_st=foreign_keys_eicu.txt
    table_func_file=tables_func_eicu.json
    dataset_path=
    valid_original=
else
  table_st=tables.txt
  foreign_st=foreign_keys.txt
  table_func_file=tables_func_mimic.json
  dataset_path=
  valid_original=
fi

echo dataset path $dataset_path
echo valid path $valid_original
echo table $table_st
echo foreign key $foreign_st
inference_stage=1


for (( i=1; i<3; i++ ))
do
  echo "Count: $i"
  inference_stage=$i
  echo "inference stage $inference_stage"
  
  if [ "$inference_stage" = 1 ]; then
      output_path=./function_call_output.txt
      pred_table_path=''
  else
    pred_table_path=./function_call_output.txt
    output_path=./output.json
  fi

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
      --pred_table_path $pred_table_path > inference_logs.txt
      
done
