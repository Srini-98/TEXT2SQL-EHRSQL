import json

if __name__ == "__main__":
    with open("valid_mimic.json" , "r") as f:
        data = json.load(f)
    
    output = []
    for i in data:
        dic = {}
        dic['question'] = i['question']
        dic['query'] = i['query']
        dic['id'] = i['id']
        output.append(dic)
    

    with open("./processed_valid_mimic.json" , "w") as g:
        json.dump(output , g)