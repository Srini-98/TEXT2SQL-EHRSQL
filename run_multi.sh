
nohup torchrun --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    train_multifunction.py \
    --model_type "mistral" \
    --model_path "./Mistral-7B-v0.1" \
    --dataset "./train_data_multi_tool_mimic_final.json" \
    --output_dir "./mistral_mimic_multi_function_1e-5/" \
    --learning_rate  1e-5 --dataset_name "mimic" > fsdp_mistral_mimic_multitool.txt