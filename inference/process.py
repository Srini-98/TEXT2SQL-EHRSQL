import json

if __name__ == "__main__":
    with open("valid.json" , "r") as f:
        data = json.load(f)
    
    output = []
    for i in data:
        dic = {}
        dic['question'] = i['question']
        dic['query'] = i['query']
        dic['id'] = i['id']
        output.append(dic)
    

    with open("./valid_processed.json" , "w") as g:
        json.dump(output , g)