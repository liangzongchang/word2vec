#!/usr/bin/env python
# -*- coding: utf-8 -*-
import word2vec
import jieba
from gensim.models import word2vec
import gensim
import logging
# 模型训练，生成词向量
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# sentences = word2vec.Text8Corpus('软件_2018-05-01_2018-05-04')  # 加载语料
# model = gensim.models.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
# model.save('软件_2018-05-01_2018-05-04'+'_model')
# model.wv.save_word2vec_format('软件_2018-05-01_2018-05-04'+'_model' + ".bin", binary=True)   # 以二进制类型保存模型以便重用
# print("model train complete!")
#
# model_1 = word2vec.Word2Vec.load('软件_2018-05-01_2018-05-04_model.bin')
# # 计算某个词的相关词列表
# y2 = model_1.most_similar("再说", topn=3)  # 10个最相关的
# print(u"和再说最相关的词有：\n")
# for item in y2:
#     print(item[0], item[1])
# print("-------------------------------\n")


# 用 word2vec 进行训练
sentences=word2vec.Text8Corpus('软件_2018-05-01_2018-05-04')
# 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
model=word2vec.Word2Vec(sentences,min_count=1, size=200)
print(model["再说"])
# y2=model.similarity(u"不错", u"好吃") #计算两个词之间的余弦距离
# print(y2)
model.wv.save_word2vec_format("软件_2018-05-01_2018-05-04_model.bin",binary=True)
model= gensim.models.KeyedVectors.load_word2vec_format("软件_2018-05-01_2018-05-04_model.bin", binary=True)
for i in model.most_similar(u"再说"): #计算余弦距离最接近“滋润”的10个词
    print(i[0],i[1])
import os
if not os.path.exists('path'):
    os.makedirs('path')