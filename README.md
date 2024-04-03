# 基于ChatGLM2-6B的电力QA系统
[第五届全球校园人工智能算法精英大赛—【算法挑战赛】电力大模型问答挑战赛](http://bdc.saikr.com/vse/48006)

利用ChatGLM2-6b实现电力领域的QA，其中knowledge中的文件均是从网上搜集和赛方提供，其中题库占大部分，主要来自百度题库。

------

## 1.解题思路

我们通过构建合适的指令，让大模型对相应的题目进行作答。同时收集电子、物理、化学、数学等学科的背景知识以及注册电气工程师的参考资料和题库，利用langchain框架，先以检索的方式检索出和当前问题相关的知识，再作为背景知识输入到模型，帮助模型进行作答。

## 2.代码说明

### 2.1 代码文件结构
```
├── document_loaders
├── knowledge
├── testsplitter
├── test.py
├── setup.py
├── predict.py
├── question.json
├── requirements.txt
└── README.md
```

### 2.2 代码文件说明

``` 
* test.py                       模型QA文件   
* setup.py                      数据格式转换文件   
* predict.py                    结果文件转换文件    
* knowledge                     使用的知识库
* requirements.txt              运行环境要求    
* question.json                 进行QA问答的数据集          
```

## 3.运行说明

### 3.1运行环境

* python == 3.10.12
```
# 1. 安装依赖
$ pip install -r requirements.txt
```

### 3.2模型下载
如需在本地或离线环境下运行本项目，需要首先将项目所需的模型下载至本地，通常开源 LLM 与 Embedding 模型可以从 HuggingFace 下载。
以本项目中默认使用的 LLM 模型 [THUDM/ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) 与 Embedding 模型 [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base) 为例：

下载模型需要先安装 Git LFS ，然后运行:
```
$ git lfs install
$ git clone https://huggingface.co/THUDM/chatglm3-6b
$ git clone https://huggingface.co/moka-ai/m3e-base
```

### 3.3运行步骤
```
# 1. 运行setup.py文件，对question.json文件格式进行调整
$ python setup.py

# 2. 运行test.py文件，对调整后的问答数据集
$ python test.py

# 3. 运行predict.py文件，将生成的结果文件格式调整为json格式
$ python predict.py
```

## 4.结果与改进
模型性能特别依赖数据集质量，其中题库是提高性能的key point，其中我们的数据集文件在格式转化的过程中存在一点问题（缺少了部分单选题以及大量的多选题和问答题的数据，最终止步于赛区二等奖），因此之后可以进行额外补充。

根据其他队伍的经验和信息，对模型进行预训练，其效果并不显著，不过可以考虑PEFT和RAG结合的方法提高模型性能，其中PEFT可以尝试LoRA、P-Tuning和QLoRA等微调方法，这也是大部分取得不错结果的队伍采用的策略。

感兴趣的话可以选择调用其他的开源 LLM 和 Embedding 模型，更大的模型效果往往更突出，例如Baichuan2-13B、Qwen1.5-14B。

以上改进建议供大家参考，欢迎大家尝试和改进提升，分享成果。

## 参考项目
https://github.com/chatchat-space/Langchain-Chatchat
