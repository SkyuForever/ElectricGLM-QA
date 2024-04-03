# 基于ChatGLM2-6B的电力QA系统
第五届全球校园人工智能算法精英大赛—【算法挑战赛】电力大模型问答挑战赛

[竞赛链接](http://bdc.saikr.com/vse/48006)

利用ChatGLM2-6b实现电力领域的QA，其中knowledge中的文件均是从网上搜集和赛方提供，其中题库占大部分，主要来自百度题库。

------

## 1.解题思路

我们通过构建合适的指令，让大模型对相应的题目进行作答。同时收集电子、物理、化学、数学等学科的背景知识以及注册电气工程师的参考资料和题库，利用langchain框架，先以检索的方式检索出和当前问题相关的知识，再作为背景知识输入到模型，帮助模型进行作答。

## 2.代码说明

### 2.1 代码文件结构
```
├── electric
│   ├── test.py
│   ├── setup.py
│   ├── predict.py
│   ├── knowledge_all
│   ├── M3e
│   ├── knowledge_base
│   ├── document_loaders
│   ├── testsplitter
│   ├── question.json
│   ├── requirements.txt
│   └── dataset
├── ChatGLM2-6b
└── README.md
```

### 2.2 代码文件说明

1、 electric部分  
``` 
* test.py                       模型QA文件   
* setup.py                      数据格式转换文件   
* predict.py                    提交文件转换文件    
* M3e 文件夹                     分词模型
* knowledge_all 文件夹           使用的知识库
* knowledge_base 文件夹          部分知识库
* requirements.txt              运行环境要求    
* question.json                 进行QA问答的数据集          
```

## 3.运行说明

### 3.1运行环境

* python == 3.10.12
```
# 1. 安装依赖
> pip install -r requirements.txt
```

### 3.2运行步骤
```
# 1. 运行setup.py文件，对question.json文件格式进行调整
> python electric/setup.py

# 2. 运行test.py文件，对调整后的问答数据集
> python test.py

# 3. 运行predict.py文件，将生成的结果文件格式调整为json格式
> python predict.py
```

## 4.结果

<div align=center>

| 提交批次 | 得分 |
| :----:| :----: |
| 1 | 35.269 |
| 2 | 35.269 |

</div>
