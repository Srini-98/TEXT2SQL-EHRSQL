input_path=$1
output_file=$2
inference_result_path=$(dirname $input_path)
input_file=$(basename $input_path)
# output_file=

python eval/abstain_with_entropy.py \
    --inference_result_path $inference_result_path \
    --input_file $input_file \
    --output_file $output_file \
    --threshold -1

    
