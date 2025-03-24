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





#football dolphins
datasets = ["football","dolphins","polbooks"]
dataset = datasets[0]
f1 = "../dataset/"+dataset+"/nodes.txt"
f2 = "../dataset/"+dataset+"/G.txt"
truthfile = "../dataset/"+dataset+"/groundTruth.csv"


list_true = GPTLCD.read_csv(truthfile)



G_neighbors={}
G = GPTLCD.read(f1,f2)
G_neighborslist_1 = []





ns1 = True  #进行节点补充
ns2 = False #不进行节点补充
iteration = 2 #迭代次数
HaveSK = True #有SK的图编码
WithoutSK = False #无SK的图编码
promptselector = 5 #1-zeroshot 2-fewshot 3-cot 4-bag 5-nsg


alllist = [1]


for seed in alllist:
    seed_list = GPTLCD.gpt_communityexpansion(seed,G,ns1,iteration,HaveSK,promptselector)
    for j in range(len(list_true)):
        flag = j
        if seed in list_true[j]:
            break
    a, b, c, d = GPTLCD.eval_scores(seed_list, list_true[flag])

    print("当前结果" + str(seed_list) +" F1:" + str(c)+" Jaccard:"+str(d))

