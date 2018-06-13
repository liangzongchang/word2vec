#!/usr/bin/env python
# -*- coding: utf-8 -*-
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




def cut_txt(old_file):
    cut_file = old_file.split('.')[0] + '_cut.txt'

    try:
        fi = open(old_file, 'r', encoding='utf-8')
    except BaseException as e:  # 因BaseException是所有错误的基类，用它可以获得所有错误类型
        print(Exception, ":", e)    # 追踪错误详细信息

    text = fi.read()  # 获取文本内容
    new_text = jieba.cut(text, cut_all=False)  # 精确模式
    print(new_text)     #一个生成器
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')     # 去掉标点符号
    fo = open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)
    return cut_file

def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名
    import logging
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = word2vec.Word2Vec(sentences, size=400,window=5,min_count=5,workers=4)  # 训练skip-gram模型; 默认window=5
    #model.save('jinyong')
    model.wv.save_word2vec_format(save_model_file + ".bin", binary=True)   # 以二进制类型保存模型以便重用
    return model,str(save_model_file + ".bin")

# outfile = cut_txt('all.txt')
model0,outmodel = model_train('cut_wiki.txt', 'wiki2')
model = gensim.models.KeyedVectors.load_word2vec_format(outmodel, binary=True)
#model = word2vec.Word2Vec.load('维基百科语料/wiki.zh.text.model')
#model.wv.save_word2vec_format('wiki' + ".bin", binary=True)  # 以二进制类型保存模型以便重用
#model0 = gensim.models.KeyedVectors.load_word2vec_format('wiki.bin', binary=True)

for i in model0.most_similar(u"电话"): #计算余弦距离最接近“滋润”的10个词
    print(i[0],i[1])
