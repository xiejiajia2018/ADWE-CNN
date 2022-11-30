# CLE4ATE
[Context-Aware Dynamic Word Embeddings For
Aspect Term Extraction](submitted to IEEE Transactions on Affective Computing and Affective Language Resources). 

## Data (师兄记得数据放上去啊)
[[Laptop](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)]
[[Restaurant 16](https://github.com/leekum2018/CLE4ATE/tree/main/Restaurants16_flat)]:

## Requirements (师兄记得填一下)
* pytorch=1.3.1
* python=3.7.5
* transformers=2.3.0
* dgl=0.5

## Steps to Run Code*  (师兄记得写一下)
- ### Step 1: 
Download official datasets and official evaluation scripts.
We assume the following file names.
SemEval 2014 Laptop (http://alt.qcri.org/semeval2014/task4/):
```
semeval/Laptops_Test_Data_PhaseA.xml
semevalLaptops_Test_Gold.xml
semeval/eval.jar
```
SemEval 2016 Restaurant (http://alt.qcri.org/semeval2016/task5/)
```
semeval/EN_REST_SB1_TEST.xml.A
semeval/EN_REST_SB1_TEST.xml.gold
semeval/A.jar
```

- ### Step 2: 
Download pre-trained model weight [[BERT-PT](https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/pytorch-pretrained-bert.md)], and place these files as:
```
bert-pt/bert-laptop/
bert-pt/bert-rest/
```
you can also specify the address of these files in config.json.

- ### Step 3: 
Train and evaluate:
```
sh train.sh
```

## Baselines

Kindly note that for reproducing the results of the following baselines, we use anaconda to create a new environment for each paper following the corresponding readme of their codes.

1. DE-CNN [[paper](https://aclanthology.org/P18-2094/)] [[code](https://github.com/howardhsu/DE-CNN)] [[checkpoints](https://drive.google.com/drive/folders/1HV2uc_4KzCp4YgrcJJyjPjKOuxqEJ9Hh?usp=share_link)]
2. Seq4Seq [[paper](https://www.aclweb.org/anthology/P19-1344.pdf)] [[code](https://github.com/madehong/Seq2Seq4ATE)] [[checkpoints](https://drive.google.com/drive/folders/1NKvn_OGj6sFz6M7qQKrIzQKx1LBuXEj3?usp=share_link)]
3. MT-TSMSA [[paper](https://aclanthology.org/2021.naacl-main.145/)] [[code](https://github.com/fengyh3/TSMSA)] [[checkpoints](https://drive.google.com/drive/folders/1zGoTskFcDp_Aue8E2244ROdrHJPRQIui?usp=share_link)]
4. CL-BERT [[paper](https://aclanthology.org/2020.coling-main.73.pdf)] [[code](https://github.com/leekum2018/CLE4ATE)] [[checkpoints](https://drive.google.com/drive/folders/1wE9c5i8Y6PBXZy0RQK-5NpCv4sE5tx9D?usp=share_link)]
