
model_path=$1
learning_rate=$2
dataset_path=$3
output_dir=$4
model_name=$5

nohup torchrun --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train.py \
    --model_path $model_path \
    --learning_rate $learning_rate \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --model_name $model_name > fsdp_logs.txt