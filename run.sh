
model_path=$1
learning_rate=$2
dataset_path=$3
output_dir=$4
model_name=$5
dataset_name=$6

nohup torchrun --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    train.py \
    --model_path $model_path \
    --dataset $dataset_path \
    --output_dir $output_dir \
    --model_type $model_name \
    --learning_rate  $learning_rate  --dataset_name $dataset_name> fsdp_llama_eicu_prof_prompt.txt