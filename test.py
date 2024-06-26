import os
import glob
import re
import json
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import FAISS
from textsplitter import ChineseTextSplitter
from langchain.document_loaders import PyPDFLoader

folder_path = 'knowledge'#知识库路径
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
filepaths = [os.path.join(folder_path, f) for f in file_names]
docs=[]

for file in filepaths:
    if file.lower().endswith(".md"):
        loader = UnstructuredFileLoader(file, mode="elements")
        docs += loader.load()
        
embeddings=SentenceTransformerEmbeddings(model_name="M3e")#embedding模型路径
vector_stroe=FAISS.from_documents(docs,embeddings)

tokenizer = AutoTokenizer.from_pretrained("ChatGLM2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("ChatGLM2-6b", trust_remote_code=True).bfloat16().cuda()

choices = ["A", "B", "C", "D"]
choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]

def build_prompt(doc,text):
    return "[Round {}]\n\n基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。\n\n已知信息：{}\n\n问：{}\n\n答：".format(1, doc, text)

extraction_prompt = '综上所述，ABCD中正确的选项是：'
answer=[]
answer1=[]
gen_kwargs = {"num_beams": 5, "do_sample": False, "top_p": 0.7, "max_length": 2048,
                      "temperature": 0.95, "logits_processor": None}
with torch.no_grad():
    for entry in glob.glob("electric/question.jsonl", recursive=True):
        dataset1 = []
        dataset2 = []
        dataset3 = []
        with open(entry, encoding='utf-8') as file:
            for line in file:
                line=json.loads(line)
                if line['type']=='单选':
                    dataset1.append(line)
                elif line['type']=='多选':
                    dataset2.append(line)
                else:
                    dataset3.append(line)
        dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=8)
        for batch in tqdm(dataloader1):
            context=[]
            texts = batch["inputs_pretokenized"]
            docs=[vector_stroe.similarity_search(text) for text in texts]
            for doc0 in docs:
                context.append([doc.page_content for doc in doc0])
            # queries = [build_prompt(query) for query in texts]
            queries = [build_prompt(doc,query) for doc,query in zip(context,texts)]
            inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512,num_beams=3,repetition_penalty=0.8,num_beam_groups=3,diversity_penalty=0.8)

            intermediate_outputs = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = tokenizer.decode(output)
                intermediate_outputs.append(response)
            answer_texts = [text + intermediate + "\n" + extraction_prompt for text, intermediate in
                            zip(texts, intermediate_outputs)]
            input_tokens = [build_prompt(doc,answer_text) for doc,answer_text in zip(context,answer_texts)]
            inputs = tokenizer(input_tokens, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model(**inputs, return_last_logit=True)
            logits = outputs.logits[:, -1]
            logits = logits[:, choice_tokens]
            preds = logits.argmax(dim=-1)
            for i in preds.cpu():
                answer.append(choices[i])
            
        
        extraction_prompt = '综上所述，ABCD中正确的选项是：'
        dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=8)
        for batch in tqdm(dataloader2):    
            # 处理多选题的逻辑
            context=[]
            texts = batch["inputs_pretokenized"]
            docs=[vector_stroe.similarity_search(text) for text in texts]
            for doc0 in docs:
                context.append([doc.page_content for doc in doc0])
            queries = [build_prompt(doc,query) for doc,query in zip(context,texts)]
            # queries = [build_prompt(query) for query in texts]
            inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512,num_beams=3,repetition_penalty=0.8,num_beam_groups=3,diversity_penalty=0.8)
            
            intermediate_outputs = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = tokenizer.decode(output)
                intermediate_outputs.append(response)
                
            answer_texts = [text + intermediate + "\n" + extraction_prompt for text, intermediate in
                            zip(texts, intermediate_outputs)]
            
            input_tokens = [build_prompt(doc,answer_text) for doc,answer_text in zip(context,answer_texts)]
            inputs = tokenizer(input_tokens, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model(**inputs, return_last_logit=True)
            logits = outputs.logits[:, -1]
            logits = logits[:, choice_tokens1]
            preds = logits.argmax(dim=-1)
            s=''
            time=0
            for i in preds.cpu():
                if choices[i]=='A':
                    s+=chr(time+ord('A'))
                time+=1
                if (time==4):
                    answer.append('、'.join(s))
                    s=''
                    time=0

        dataloader3 = torch.utils.data.DataLoader(dataset3, batch_size=8)
        for batch in tqdm(dataloader3):      
            # 处理问答题的逻辑
            context=[]
            texts = batch["inputs_pretokenized"]
            docs=[vector_stroe.similarity_search(text) for text in texts]
            for doc0 in docs:
                context.append([doc.page_content for doc in doc0])
            # queries = [build_prompt(query) for query in texts]
            queries = [build_prompt(doc,query) for doc,query in zip(context,texts)]
            inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512,num_beams=3,repetition_penalty=0.8,num_beam_groups=3,diversity_penalty=0.8)
            generated_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            generated_responses = [re.search(r'答：(.*)', text, re.DOTALL).group(1).strip() for text in generated_responses]
            for i in generated_responses:
                answer.append(i)

with open('electric/output.jsonl', 'w', encoding='utf-8') as output_file:
    for i in range(len(answer)):
        output_data={
            "id": i,
            "predict":answer[i]
        }
        json.dump(output_data, output_file,ensure_ascii=False)
        output_file.write('\n')
