import json

# 假设你有多个数据，存储在一个列表中
with open('question.json','r',encoding='utf8') as file:
    data_list=json.load(file)

# 将数据写入JSON Lines文件
with open('question.jsonl', 'w',encoding='utf8',) as file:
    for data in data_list:
        question_text = data["question"]
        
        correct_answer = 0
        if data['type']=='单选':
            choices = [data["A"], data["B"], data["C"], data["D"]]
            output_data = {
                "id": data["id"],
                'type':data['type'],
                "inputs_pretokenized": f"{question_text}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
                "choices_pretokenized": [" A", " B", " C", " D"],
                "label": correct_answer,
                "targets_pretokenized": [""]
            }

            json.dump(output_data, file,ensure_ascii=False)
            file.write('\n')
        elif data['type']=='多选':
            choices = [data["A"], data["B"], data["C"], data["D"]]

            output_data = {
                "id": data["id"],
                'type':data['type'],
                "inputs_pretokenized": f"{question_text}\nA. {choices[0]}\nB. 不存在正确答案\nC. 不存在正确答案\nD. 不存在正确答案",
                "choices_pretokenized": [" A", " B", " C", " D"],
                "label": [0,1],
                "targets_pretokenized": [""]
            }

            json.dump(output_data, file,ensure_ascii=False)
            file.write('\n')

            output_data = {
                "id": data["id"],
                'type':data['type'],
                "inputs_pretokenized": f"{question_text}\nA. {choices[1]}\nB. 不存在正确答案\nC. 不存在正确答案\nD. 不存在正确答案",
                "choices_pretokenized": [" A", " B", " C", " D"],
                "label": [0,1],
                "targets_pretokenized": [""]
            }

            json.dump(output_data, file,ensure_ascii=False)
            file.write('\n')

            output_data = {
                "id": data["id"],
                'type':data['type'],
                "inputs_pretokenized": f"{question_text}\nA. {choices[2]}\nB. 不存在正确答案\nC. 不存在正确答案\nD. 不存在正确答案",
                "choices_pretokenized": [" A", " B", " C", " D"],
                "label": [0,1],
                "targets_pretokenized": [""]
            }

            json.dump(output_data, file,ensure_ascii=False)
            file.write('\n')

            output_data = {
                "id": data["id"],
                'type':data['type'],
                "inputs_pretokenized": f"{question_text}\nA. {choices[3]}\nB. 不存在正确答案\nC. 不存在正确答案\nD. 不存在正确答案",
                "choices_pretokenized": [" A", " B", " C", " D"],
                "label": [0,1],
                "targets_pretokenized": [""]
            }

            json.dump(output_data, file,ensure_ascii=False)
            file.write('\n')
        
        else:
            output_data = {
                "id": data["id"],
                'type':data['type'],
                "inputs_pretokenized": f"{question_text}",
                "targets_pretokenized": [""]
            }

            json.dump(output_data, file,ensure_ascii=False)
            file.write('\n')
