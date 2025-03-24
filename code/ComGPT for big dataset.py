import multiprocessing
import math
import copy
import time
import os
import openai
import re
import random
import csv
import GPTLCD
from typing import List, Set, Union


##主函数
datasets = ["amazon", "dblp"]
dataset = datasets[0]
filename = "../dataset/"+dataset+"/"+dataset+".txt"
G = GPTLCD.read_bigdataset(filename)
##计算F1
truthfile = "../dataset/"+dataset+"/realdata.txt"
list_true = GPTLCD.read_truthbigdataset(truthfile)




G_neighbors={}
G_neighborslist_1 = []

alllist = []


sum = 0
length = 0
i = 0
ns1 = True  #进行节点补充
ns2 = False #不进行节点补充
iteration = 2
HaveSK = True
WithoutSK = False
promptselector = 5 #1-zeroshot 2-fewshot 3-cot 4-bag 5-nsg

for seed in alllist:
    seed_list = GPTLCD.gpt_communityexpansion(seed,G,ns1,iteration,HaveSK,promptselector)


    realcommunity = []
    count = 0
    for j in range(len(list_true)):
        if seed in list_true[j]:
            set1 = set(realcommunity)
            set2 = set(list_true[j])
            union_set = set1 | set2
            realcommunity = list(union_set)
            count = count + 1
    a, b, c, d = GPTLCD.eval_scores(seed_list, realcommunity)
    # print(count)
    # print("GPT-LCD:",seed_list)
    # print("Ground-truth",realcommunity)
    sum = sum + c
    length = length + 1
    print("当前结果" + str(seed_list) + " F1:" + str(c) + " Jaccard:" + str(d))
