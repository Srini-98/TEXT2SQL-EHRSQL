dataset_type=$4
data_input_file=$1
pred_input_file=$2
output_file=$3
#set if confition for dataset_type mimic3 or eicu
if [ $dataset_type == "mimic3" ]
then
    db_path=database/mimic_iii.db
    data_file=$data_input_file

elif [ $dataset_type == "eicu" ]
then
    db_path=database/eicu.db
    data_file=$data_input_file
fi

echo $db_path
echo $data_file


#./../dataset/ehrsql/eicu/valid.json
pred_file=$pred_input_file

python eval/evaluate.py \
    --db_path  $db_path \
    --data_file  $data_file \
    --pred_file $pred_file \
    #--out_file $output_file
