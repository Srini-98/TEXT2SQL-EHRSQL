import argparse
import json


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--pred_file" , type=str , required=True)
    args.add_argument("--output_file" , type=str , required=True)
    args = args.parse_args()
    with open("./../dataset/ehrsql/mimic_iii/valid.json" , "r") as f:
        data = json.load(f)
    
    with open(args.pred_file , "r") as g:
        data_pred = json.load(g)
    
    dic = {}
    for i , j in data_pred.items():
        dic[i] = j['pred']
    
    with open(args.output_file , "w") as l:
        json.dump(dic , l)
   