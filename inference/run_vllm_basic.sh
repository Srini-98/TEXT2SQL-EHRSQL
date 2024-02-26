peft_path=" "
top_p=1
temperature=0


device=0
output_sub_path=./

echo "starting inference"
data_used=$1
sub_data_mimic=$2
count=3

if [ "$sub_data_mimic" = "eicu" ]; then
    dataset_path=$3
    valid_original=$4
    table_st=tables_eicu.txt
    foreign_st=foreign_keys_eicu.txt
else
  dataset_path=$3
  valid_original=$4
  table_st=tables.txt
  foreign_st=foreign_keys.txt
fi

echo dataset path $dataset_path
echo valid path $valid_original
echo table $table_st
echo foreign key $foreign_st

model_path=$5


output_path=./output.json
echo  $model_path

CUDA_VISIBLE_DEVICES=$device nohup python ./inference.py \
    --model_name $model_path \
    --top_p $top_p \
    --temperature $temperature \
    --dataset_path $dataset_path \
    --data_used  $data_used \
    --output_path $output_path \
    --valid_original $valid_original \
    --foreign_st $foreign_st \
    --table_st $table_st > vllm_inference_logs.txt