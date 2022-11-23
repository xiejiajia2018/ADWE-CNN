# CLE4ATE
[Context-Aware Dynamic Word Embeddings For
Aspect Term Extraction](submitted to IEEE Transactions on Affective Computing and Affective Language Resources). 

## Data
[[Laptop](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)]
[[Restaurant 16](https://github.com/leekum2018/CLE4ATE/tree/main/Restaurants16_flat)]:

## Requirements (??)
* pytorch=1.3.1
* python=3.7.5
* transformers=2.3.0
* dgl=0.5

## Steps to Run Code
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

1. DE-CNN [[paper](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[code](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[checkpoints](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)]
2. Seq4Seq [[paper](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[code](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[checkpoints](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)]
3. MT-TSMSA [[paper](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[code](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[checkpoints](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)]
4. CL-BERT [[paper](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[code](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)] [[checkpoints](https://github.com/leekum2018/CLE4ATE/tree/main/Laptops_flat)]
