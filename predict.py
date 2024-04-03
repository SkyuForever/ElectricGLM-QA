import json
data_item=[]
with open('electric/output.jsonl','r',encoding='utf8') as file:
    for line in file:
        data_item.append(json.loads(line))

output=[]
for data in data_item:
    if data['predict']=='':
        data['predict']='A';
    json_data={
        'ID':data['id'],
        'answer':data['predict']
    }
    output.append(json_data)

with open('electric/_result.json','w',encoding='utf8') as output_file:
    json.dump(output,output_file,ensure_ascii=False,indent=2)