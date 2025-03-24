# -*- coding: utf-8 -*-

import multiprocessing
import math
import copy
import time
import os
import openai
import re
import random
import csv
import time
import requests
import json
from typing import List, Set, Union
import tiktoken

def read(file1,file2):
    """
    读取两个文件中的数据以构建全局图。

    参数:
    file1 (str): 包含节点列表的文件路径。该文件中的每一行代表图中的一个节点。
    file2 (str): 包含边列表的文件路径。该文件中的每一行代表两个节点之间的边，节点由空格分隔的整数表示。

    返回值:
    dict: 一个表示图 G 的字典，其中键是节点标识符（整数），值是邻居节点（整数）列表。
    """
    G = {}
    with open(file1) as f:
        lines = f.readlines()
        for line in lines:
            dict = {int(line): []}
            G.update(dict)
    with open(file2) as ef:
        elines = ef.readlines()
        for line in elines:
            cureline = line.strip().split(" ")
            G[int(cureline[0])].append(int(cureline[1]))
            G[int(cureline[1])].append(int(cureline[0]))
    return (G)

def read_csv(file_path):
    """
    从CSV文件中获取真实分布（如football数据集）。

    参数:
    file_path (str): CSV文件的路径，文件中每一行表示一个节点或社区的分布。

    返回值:
    list: 一个列表，列表中的每个元素都是一个子列表，表示每个社区中的节点编号。
    """
    list_true = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            list_true.append(row)
    list_truecomm = []
    for i in range(int((len(list_true) + 1) / 2)):
        list_true_son = [int(x) for x in list_true[i * 2]]
        list_truecomm.append(list_true_son)
    return list_truecomm

def read_bigdataset(filename):
    """
    读取数据集（如 Amazon 或 DBLP），以构建全局无向图。

    参数:
    filename (str): 文件路径，文件中的每一行表示两个节点之间的边，节点由空格分隔的整数表示。

    返回值:
    dict: 一个表示图的字典，键是节点的标识符（整数），值是相邻节点（整数）列表。
    """

    graph = {}
    with open(filename, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            if node1 not in graph:
                graph[node1] = []
            if node2 not in graph:
                graph[node2] = []
            graph[node1].append(node2)
            graph[node2].append(node1)  # 因为是无向图，所以两个节点都要互相连接
    return graph

def read_truthbigdataset(filename):
    """
    读取数据集文件以获取真实分布（如 Amazon 数据集）。

    参数:
    filename (str): 文件路径，文件中的每一行表示一个由节点组成的序列，节点由空格分隔的整数表示。

    返回值:
    list: 一个列表，列表中的每个元素都是一个子列表，表示一个节点序列（例如，社区或簇）。
    """
    sequences = []
    with open(filename, 'r') as file:
        for line in file:
            sequence = list(map(int, line.strip().split()))  # 去除每行两端的空白字符，然后按空格分割，并转换为整数列表
            sequences.append(sequence)
    return sequences

def getneighbors1(seed_list,G):
    """
    输入当前社区和图结构，获取当前社区的一阶邻居节点列表。

    参数:
    seed_list (list): 当前社区中的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    list: 当前社区的一阶邻居节点列表，不包含当前社区中的节点。
    """
    G_neighborslist_1 = []
    temp = []
    for seed in seed_list:
        temp = G[seed]
        for son in temp:
            G_neighborslist_1.append(son)
    G_neighborslist_1 = set(G_neighborslist_1)
    G_neighborslist_1 = [x for x in G_neighborslist_1 if x not in seed_list]
    return G_neighborslist_1

def getneighbors2(seed_list,G):
    """
    输入当前社区和图结构，获取当前社区的二阶邻居节点列表。

    参数:
    seed_list (list): 当前社区中的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    list: 当前社区的二阶邻居节点列表，不包含当前社区中的节点。
    """
    G_neighborslist_1 = getneighbors1(seed_list,G)
    G_neighborslist_2 = []
    temp = []
    for node in G_neighborslist_1:
        temp = G[node]
        for son in temp:
            G_neighborslist_2.append(son)
    G_neighborslist_2 = set(G_neighborslist_2)
    G_neighborslist_2 = [x for x in G_neighborslist_2 if x not in seed_list]
    return G_neighborslist_2

def computeM(community_list,G):
    """
    输入当前社区和图结构，计算当前社区的M值。

    参数:
    community_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    float: 当前社区的M值。
    """
    community_set = set(community_list)
    ein = 0
    eout = 0
    for node in community_set:
        nodelist = G[node]
        for nodee in nodelist:
            if nodee in community_set:
                ein = ein + 1
            else:
                eout = eout + 1
    ein = ein / 2
    if(eout==0):
        return -1
    else:
        M = ein / eout
        M = M*len(community_set)/len(community_list)
        return M

def getcandidate(d,length):
    """
    输入保存M值的字典和参数k，获得M值最大的k个节点。

    参数:
    d (dict): 一个字典，键是节点，值是该节点对应的M值。
    length (int): 需要返回的节点数量，即前k个M值最大的节点。

    返回值:
    list: 一个列表，包含M值最大的k个节点的键。
    """
    sorted_dict = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_dict[:length]]

def StorageM(G_neighborslist_1,community_list,G):
    """
    输入当前社区、一阶邻居节点列表和图结构，获取当前社区邻居节点中M值增量（DeltaM）大于0的节点。

    参数:
    G_neighborslist_1 (list): 当前社区的一阶邻居节点列表。
    community_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    dict: 一个字典，包含M值增量（DeltaM）大于0的节点，键为节点，值为该节点加入社区后的新M值。
    """
    mdict = {}
    M1 = computeM(community_list,G)
    for node in G_neighborslist_1:
        Mtemp = copy.deepcopy(community_list)
        Mtemp.append(node)
        M2 = computeM(Mtemp,G)
        M3 = M2-M1
        if M3 > 0 :#threshold
            mdict.update({node: M2})
    return mdict

def StorageM2(G_neighborslist_1,community_list,G):
    """
    输入当前社区、一阶邻居节点列表和图结构，获取当前社区每一个邻居节点加入后的M值。

    参数:
    G_neighborslist_1 (list): 当前社区的一阶邻居节点列表。
    community_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    dict: 一个字典，包含每个邻居节点在加入社区后的M值，键为邻居节点，值为该节点加入社区后的新M值。
    """
    mdict = {}
    M1 = computeM(community_list,G)
    for node in G_neighborslist_1:
        Mtemp = copy.deepcopy(community_list)
        Mtemp.append(node)
        M2 = computeM(Mtemp,G)
        mdict.update({node: M2})
    return mdict

def getGneighbors(seedlist,N,G):
    """
    输入当前社区、一阶邻居节点列表和图结构，获取当前社区和邻居节点的邻接表。

    参数:
    seedlist (list): 当前社区的节点列表。
    N (list): 一阶邻居节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    dict: 一个字典 `G_neighbors`，表示当前社区和邻居节点的邻接表，键为节点，值为与该节点相邻的节点列表。
    """
    G_neighbors = {}
    for node in seedlist:
        nodedict = {int(node): []}
        G_neighbors.update(nodedict)
    for node in N:
        nodedict = {int(node): []}
        G_neighbors.update(nodedict)
    for node in seedlist:
        for nodee in G[node]:
            G_neighbors[node].append(nodee)
    for node in N:
        for nodee in G[node]:
            G_neighbors[node].append(nodee)
    return G_neighbors

def evalcandidate_m(seed_list,G,K):
    """
    输入当前社区、一阶邻居和参数K，获取Delta M大于0的前K个节点。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    K (int): 需要返回的节点数量，即Delta M大于0的前K个节点。

    返回值:
    list: 包含Delta M大于0的前K个节点的列表。
    """
    G_neighborslist_1 = getneighbors1(seed_list, G)  # 社区一阶邻居
    mdict = StorageM(G_neighborslist_1, seed_list,G)
    candidate = getcandidate(mdict,K)
    return candidate

def Mpatch(seed_list,G,K):
    """
    输入当前社区、一阶邻居和参数K，获取用于节点补充的节点。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    K (int): 需要返回的节点数量，即前K个潜在补充的节点。

    返回值:
    list: 获取用于节点补充的节点。
    """
    G_neighborslist_1 = getneighbors1(seed_list, G)  # 社区一阶邻居
    mdict = StorageM2(G_neighborslist_1, seed_list,G)
    candidate = getcandidate(mdict,K)
    return candidate

def getevalcanidate(seed_list,G, K):
    """
    输入当前社区、一阶邻居和参数K，获取节点选择的潜在节点。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    K (int): 需要返回的节点数量，即前K个潜在补充的节点。

    返回值:
    list: 包含潜在补充的前K个节点的列表。
    """
    G_neighborslist_1 = getneighbors1(seed_list, G)  # 社区一阶邻居
    candidate1 = evalcandidate_m(seed_list, G, K)

    return candidate1

def communirytostr(seed_list):
    """
    输入当前社区的节点列表，将其转换为图结构的文本描述。

    参数:
    seed_list (list): 当前社区的节点列表。

    返回值:
    str: 一个字符串，表示社区节点的文本描述，每个节点之间使用 "and" 连接。
    """
    communitystr = ""
    str1 = " and "
    count = 0
    for node in seed_list:
        nodestr = "node " + str(node)
        communitystr = communitystr + nodestr
        count = count + 1
        if count < len(seed_list):
            communitystr = communitystr + str1
    return communitystr


def GraphtoStr2(G):
    """
    输入图结构，获取Incident编码法的文本描述结果。

    参数:
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。

    返回值:
    str: 图结构的文本描述，使用Incident编码法表示。
    """
    Gstr = "G describes a graph among "
    seed_list = []
    for key,value in G.items():
        seed_list.append(key)
        strnum = str(key)+","
        Gstr = Gstr + strnum
    Gstr = Gstr + " In this graph:"
    for key,value in G.items():
        stredge = "Node "+str(key)+" is connected to nodes "
        for node in G[key]:
            if node in seed_list:#只考虑一阶节点
                if node != G[key][-1]:
                    strnode = str(node) + ","
                else:
                    strnode = str(node) + ". "
                stredge = stredge + strnode

        Gstr = Gstr + stredge
    return Gstr

def getlocalgraph(seed_list,G,K):
    """
    输入当前社区、图结构和参数K，获取当前社区和潜在节点的局部图邻接表。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    K (int): 潜在节点的数量。

    返回值:
    dict: 当前社区及其潜在节点构成的局部图的邻接表。
    """
    candidate = getevalcanidate(seed_list,G,K)
    localnode = seed_list+candidate
    G_local = {}
    for node in localnode:
        nodedict = {int(node): []}
        G_local.update(nodedict)
    for node in localnode:
        for nodee in G[node]:
            if nodee in localnode:
                G_local[node].append(nodee)
    return G_local

def getjudgegrpah(seed_list,G,i,K):
    """
    输入当前社区、图结构、分类符i和参数K，根据需求获取不同的邻接表。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    i (int): 分类符，用于确定获取邻接表的方式。
    K (int): 潜在节点的数量。

    返回值:
    dict: 生成的邻接表，根据分类符i的不同而不同。
    """
    if(i==1):#当前社区和潜在节点的邻接表
        mpatch = Mpatch(seed_list, G,K)
        localnode = seed_list + mpatch
        G_local = {}  # 只有社区节点和补充节点
        for node in localnode:
            nodedict = {int(node): []}
            G_local.update(nodedict)
        for node in localnode:
            for nodee in G[node]:
                if nodee in localnode:
                    G_local[node].append(nodee)
        return G_local
    if(i==2): # 当前社区及其一阶邻居的邻接表
        G_local2 = {}
        mpatch = Mpatch(seed_list, G,K)
        localnode2 = seed_list + mpatch

        for node in mpatch:
            for nodee in G[node]:
                if nodee not in localnode2:
                    localnode2.append(nodee)
        for node in localnode2:
            nodedict2 = {int(node): []}
            G_local2.update(nodedict2)
        for node in localnode2:
            for nodee in G[node]:
                if nodee in localnode2:
                    G_local2[node].append(nodee)
        return G_local2
    if(i==3):# 在i=2的基础上加入二阶邻居
        G_neighborslist_1 = getneighbors1(seed_list, G)
        G_neighbors2order = getneighbors2(seed_list, G)

        G_local3 = {}  # 社区节点补充节点及其相关的一阶邻居
        localnode3 = seed_list + G_neighborslist_1 + G_neighbors2order

        for node in localnode3:
            nodedict3 = {int(node): []}
            G_local3.update(nodedict3)
        for node in localnode3:
            for nodee in G[node]:
                if nodee in localnode3:
                    G_local3[node].append(nodee)
        return G_local3


def Graphencoder(seed_list,G,i,K,SK):
    """
    输入当前社区、图结构、分类符i、参数K和判断是否具有补充知识SK。获取完整的图文本。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    i (int): 分类符，用于确定编码图的方式。
    K (int): 潜在节点的数量。
    SK (bool): 是否包含补充知识的标志。

    返回值:
    str: 完整的图文本，根据分类符 `i` 和 `SK` 的不同而不同。
    """
    if(i==1): #节点选择的图编码
        #图拓扑
        candidate = getevalcanidate(seed_list, G,K)
        G_communitycandi = getlocalgraph(seed_list, G,K)
        Incident = GraphtoStr2(G_communitycandi)

        #补充知识
        connectstr = ""
        for node in candidate:
            nodelist = G[node]
            common = [element for element in seed_list if element in nodelist]
            nodestr = "Node " + str(node) + " is connected to nodes within the community: "
            for nodee in common:
                if (nodee == common[-1]):
                    nodestr = nodestr + str(nodee) + ". "
                    continue
                nodestr = nodestr + str(nodee) + ","
            nodestr2 = "Node " + str(node) + " is connected to nodes outside community: "
            out_list = [element for element in nodelist if element not in common]
            out_list = list(set(out_list) & set(candidate))
            if(out_list):
                for nodee in out_list:
                    if (nodee == out_list[-1]):
                        nodestr2 = nodestr2 + str(nodee) + ". "
                        continue
                    nodestr2 = nodestr2 + str(nodee) + ","
            else:
                nodestr2 = nodestr2 + "null."

            connectstr = connectstr + nodestr + nodestr2

        IncidentwithSK = GraphtoStr2(
            G_communitycandi) + ". Supplementary knowledge: Nodes in the current community: " + str(
            seed_list) + ". The outside nodes contain:" + str(candidate) + ". " + connectstr

        if(SK):
            return IncidentwithSK#有补充知识
        else:
            return Incident#无补充知识


    if(i==2):#节点补充的图编码
        #图拓扑
        #社区和待定及其一阶邻居
        G_judge = getjudgegrpah(seed_list, G,2,K)#1:社区及其补充节点   2：1+一阶邻居   3：社区及其二阶邻居
        graphtext_judge = GraphtoStr2(G_judge)
        #社区及其一二阶邻居
        G_judge2 = getjudgegrpah(seed_list, G,3,K)
        CN2 = G_judge2.keys()
        N2 = []
        for node in CN2:
            if node not in seed_list:
                N2.append(node)

        #补充知识
        mpath = Mpatch(seed_list, G, K)
        connectstr = ""
        for node in mpath:
            nodelist = G[node]
            common = [element for element in seed_list if element in nodelist]
            nodestr = "Node "+str(node) + " is connected to nodes within the community: "
            for nodee in common:
                if(nodee==common[-1]):
                    nodestr = nodestr + str(nodee) + ". "
                    continue
                nodestr = nodestr +str(nodee)+","
            nodestr2 = "Node "+str(node) + " is connected to nodes outside community: "
            out_list = [element for element in nodelist if element not in common]
            for nodee in out_list:
                if(nodee==out_list[-1]):
                    nodestr2 = nodestr2 + str(nodee) + ". "
                    continue
                nodestr2 = nodestr2 +str(nodee)+","
            connectstr = connectstr + nodestr + nodestr2

        graphtext_judge2 = GraphtoStr2(G_judge2) + ". Supplementary knowledge: The nodes in community contains " + str(seed_list) + ". The outside nodes contain:" +str(N2)+". "+ connectstr

        return graphtext_judge2


def instrucionstr(seed_list,G,i,K):#输入当前社区，图结构，分类符i和参数K。获取节点选择或者节点补充的指令
    """
    输入当前社区、图结构、分类符i和参数K，获取节点选择或者节点补充的指令。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    i (int): 分类符，用于确定获取指令的类型（1：节点选择，2：节点补充）。
    K (int): 潜在节点的数量。

    返回值:
    str: 对应分类符的指令字符串，用于指导节点选择或节点补充操作。
    """
    if(i == 1):#节点选择
        instruction = "You're doing local community detection. Based on the graph data and prompt, please select a node that you think is most likely to belong to the current community"+ str(seed_list) + " for community expansion. Provide a detailed explanation."# You are a person who speaks briefly. You only tell the result when answering the question, without giving the reason. You only need to output the number of selected node"

        return instruction

    if(i==2):#节点补充
        mpath = Mpatch(seed_list, G,K)

        instruction = "Please analyze whether these nodes" + str(mpath) + " should be added to the community" + str(seed_list) + ". The probability of not adding nodes is higher. But it doesn't mean you always refuse to add nodes. If you think there is a suitable node, please output its node number."


        return instruction


def prompt(i,promptselect,candidate):
    """
    输入分类符i，提示符选择符promptselect和潜在节点，获取不同的提示符。

    参数:
    i (int): 分类符，用于确定提示的类型（1：节点选择，2：节点补充）。
    promptselect (int): 提示符选择符，确定使用哪种提示策略（1：Zeroshot，2：Fewshot，3：CoT，4：BaG，5：NSG）。
    candidate (list): 潜在节点列表。

    返回值:
    str: 对应分类符和提示符选择符的提示符字符串，用于指导用户操作。
    """
    if(i == 1):#节点选择
        prompt_NSG = "Please find the node that best meets these two guides from the outside nodes " + str(candidate) + " to answer the question. Guide 1:The more an outside node is connected to other outside nodes, the higher the likelihood of its selection. Guide 2:Prioritize selecting outside nodes that are connected to multiple nodes within the community."

        prompt_fewshot = "Here are some examples for your reference:Example 1:(1)Graph data：G describes a graph among nodes a，b，c，d，f，g，h,  j.In this graph: Node a connects nodes b,d,g.Node b connects nodes a,c,d,f,h,j.Node c connects nodes b,d.Node d connects nodes a,b,c,f,g.Node f connects nodes b,d,j.Node g connects nodes a,d.Node h connects nodes b.Node j connects nodes b,f.  Supplementary knowledge: Nodes in the current community: [a,b].The outside nodes contains :[c,d,f,h,g,j].Node c is connected to nodes within the community:b. Node c is connected to nodes outside community: d. Node d is connected to nodes within the community:a, b. Node d is connected to nodes outside community: c,f,g. Node f is connected to nodes within the community:b. Node f is connected to nodes outside community: d,j. Node g is connected to nodes within the community:a. Node g is connected to nodes outside community: d. Node h is connected to nodes within the community:b. Node h is connected to nodes outside community: null.Node j is connected to nodes within the community:b. Node j is connected to nodes outside community: f.(2)Question: You're doing local community detection. Based on the graph data, please select a node that you think is most likely to belong to the current community[a,b] for community expansion.(3)Answer:Node d.  Example 2:(1)Graph data：G describes a graph among nodes a，b，c，d，e，f. In this graph: Node a connects nodes b,c,d,e.Node b connects nodes a,c.Node c connects nodes a,b,d,f.Node d connects nodes a,c.Node e connects nodes a,f.Node f connects nodes c,e.Supplementary knowledge: Nodes in the current community: [a,b,c]. The outside nodes contains [d,e,f].Node d is connected to nodes within the community:a,c. Node d is connected to nodes outside community: null.Node e is connected to nodes within the community:a. Node e is connected to nodes outside community: f.Node f is connected to nodes within the community:c. Node f is connected to nodes outside community: e.(2)Question:  You're doing local community detection. Based on the graph data, please select a node that you think is most likely to belong to the current community[a,b,c] for community expansion.(3)Answer:Node d"
        prompt_zeroshot = "null"
        prompt_cot = "Let's think step by step."
        prompt_bag = "Let's construct a graph with the nodes and edges first."

        if promptselect == 1:
            return prompt_zeroshot
        elif promptselect == 2:
            return prompt_fewshot
        elif promptselect == 3:
            return prompt_cot
        elif promptselect == 4:
            return prompt_bag
        elif promptselect == 5:
            return prompt_NSG
        else:
            print("Your promptselector is error ")
            return ""


    if(i == 2):#节点补充
        #prompt_add = "Definition of community: nodes in the same community are tightly connected,while the nodes in different communities are sparsely connected.Do not easily add nodes to the community; only add nodes to the community if you are fully confident.The more a node connects with nodes within a community, the more it can increase the community's cohesion. Conversely, the more it connects with nodes outside the community, the more it can decrease the community's cohesion.If you choose to add a node to the community, it should make the connections within the community tighter."
        prompt_add = "Definition of community: nodes in the same community are tightly connected, while the nodes in different communities are sparsely connected.The more a node connects with nodes within a community, the more it can increase the community's cohesion. Conversely, the more it connects with nodes outside the community, the more it can decrease the community's cohesion. If you choose to add a node to the community, it should make the connections within the community tighter."

        return prompt_add


def extract_number_from_string(input_string):
    """
    从输入字符串中提取第一个数字。

    参数:
    input_string (str): 输入的字符串，可能包含数字和其他字符。

    返回值:
    int 或 None: 返回匹配到的第一个数字（整数形式）。如果没有找到数字，返回 None。
    """
    match = re.search(r'\d+', input_string)
    if match:
        # 返回匹配到的第一个数字
        return int(match.group())
    else:
        # 没有找到数字
        return None


def getgpt(data):#
    """
    调用GPT API并返回生成的回复。

    参数:
    data (str): 用户输入的数据，将作为用户对话内容发送给GPT。

    返回值:
    str: GPT模型生成的回复。
    """
    #配置网络信息
    os.environ["http_proxy"] =
    os.environ["https_proxy"] =

    openai.api_key =  "yours"

    # 构建对话
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
        {"role": "user", "content": data}
    ]

    # 调用API以进行对话
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",  # 使用 GPT-3.5 Turbo 模型
        messages=messages,
    )

    # 提取助手的回复
    assistant_reply = response["choices"][0]['message']['content']
    reason = response["choices"][0]['finish_reason']

    print(assistant_reply)
    # print(reason)
    # print(response['usage'])
    return assistant_reply


def llms(seed_list,G,i,K,SK,promptselect):
    """
    输入当前社区、图、迭代次数、K、图编码方式和提示词，获得当前社区执行一次节点补充或节点选择后的社区。

    参数:
    seed_list (list): 当前社区的节点列表。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    i (int): 分类符，用于确定执行节点选择还是节点补充（1：节点选择，2：节点补充）。
    K (int): 潜在节点的数量。
    SK (bool): 是否包含补充知识的标志。
    promptselect (int): 提示符选择符，确定使用哪种提示符。

    返回值:
    list: 更新后的社区节点列表。
    """
    if (i == 1):  # 节点选择
        #获取输入给GPT的数据
        candidate = getevalcanidate(seed_list, G,K)
        localgraphnodes = seed_list + candidate
        communitystr = communirytostr(seed_list)  # 得到表述社区的字符串
        G_neighborslist_1 = getneighbors1(seed_list, G)  # 一阶邻居列表
        G_neighbors = getGneighbors(seed_list, G_neighborslist_1, G)  # 种子社区和一阶邻居组成的邻接表
        communityout = [item for item in G_neighborslist_1 if item not in seed_list]
        data = "(1)Graph data：" + Graphencoder(seed_list, G, 1, K,SK) + ".\n(2)Prompt:" + prompt(
            1,promptselect,candidate) + "\n(3)Question:" + instrucionstr(seed_list, G, 1,K)#输入给GPT的文本
        print(data)
        #reply = getgpt_agent(data)
        #与GPT的交互过程
        reply = getgpt(data)+ "The above paragraph is used to determine if a node should be added to the community and which one. If this paragraph determines that nodes can be added, please output nodes directly, otherwise output null. the scope of your answer is limited to nodes or null. please do not output anything other than nodes or null."
        reply2 = getgpt(reply)
        outnum = extract_number_from_string(reply2)

        while(len(seed_list)==1 and outnum == seed_list[0]):
            reply = getgpt(data) + "The above paragraph is used to determine if a node should be added to the community and which one. If this paragraph determines that nodes can be added, please output nodes directly, otherwise output null. the scope of your answer is limited to nodes or null. please do not output anything other than nodes or null."
            reply2 = getgpt(reply)
            outnum = extract_number_from_string(reply2)

        while (outnum not in localgraphnodes):
            reply = getgpt(data) + "The above paragraph is used to determine if a node should be added to the community and which one. If this paragraph determines that nodes can be added, please output nodes directly, otherwise output null. the scope of your answer is limited to nodes or null. please do not output anything other than nodes or null."
            reply2 = getgpt(reply)
            outnum = extract_number_from_string(reply2)

        seed_list.append(outnum)
        print("添加" + str(outnum) + "进入社区，此时社区为" + str(seed_list))

        return seed_list

    if(i == 2):#节点补充
        #获得输入给GPT的数据
        candidate = getevalcanidate(seed_list, G,K)
        judgedata = "(1)Graph data：" + Graphencoder(seed_list,G, 2,K,SK) + ".\n(2)Prompt:" + prompt(2,promptselect,candidate) +"\n(3)Question:" + instrucionstr(seed_list,G,2,K)
        print(judgedata)
        #与GPT交互
        reply = getgpt(judgedata) + "The above paragraph is used to determine if a node should be added to the community and which one. If this paragraph determines that nodes can be added, please output nodes directly, otherwise output null. the scope of your answer is limited to nodes or null. please do not output anything other than nodes or null."
        reply2 = getgpt(reply)
        outnum = extract_number_from_string(reply2)
        mpath = Mpatch(seed_list, G,K)
        if (outnum in mpath):
            print("补充节点：", outnum)
            seed_list.append(outnum)
            return seed_list
        else:
            return 0


def gptselectnodewithns(seed,G,iteration,K,SK,promptselect):
    """
    使用具有节点补充的算法，输入种子节点、图结构、迭代次数、参数K、是否使用补充知识、提示符，获得最终社区。通过多次迭代，使用GPT进行节点选择和节点补充，获得最终社区。

    参数:
    seed (int): 初始种子节点。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    iteration (int): 迭代次数。
    K (int): 潜在节点的数量。
    SK (bool): 是否包含补充知识的标志。
    promptselect (int): 提示符选择符，确定使用哪种提示策略。

    返回值:
    list: 最终的社区节点列表。
    """
    i=0
    result = []
    while(i<iteration):
        seed_list = [seed]
        seed_list11 = []
        candidate = getevalcanidate(seed_list, G, K)  #寻找潜在节点
        count = 0
        count2 = 0
        sleepcount = 0
        repeat = len(seed_list) - len(set(seed_list))
        k = len(seed_list) / 3
        stop = 0
        cands = []
        while (repeat<k and stop==0): #判断算法是否终止
            while ((len(candidate) > 0) and (repeat<k)):
                if len(candidate) == 1:
                    seed_list.append(candidate[0])
                    candidate = getevalcanidate(seed_list, G,K)
                    repeat = len(seed_list) - len(set(seed_list))
                    k = len(seed_list) / 3
                    continue

                seed_list = llms(seed_list, G, 1,K,SK,promptselect)  # 节点选择
                repeat = len(seed_list) - len(set(seed_list))
                k = len(seed_list) / 3
                candidate = getevalcanidate(seed_list, G,K)
                sleepcount = sleepcount + 1
                if (sleepcount % 10 == 0):#缓解超过访问限制
                    time.sleep(10)
            tempcand = seed_list.copy()
            cands.append(tempcand)
            # 进行节点补充
            flag2 = llms(seed_list, G, 2,K,SK,promptselect)
            if (flag2 == 0):
                stop = 1
                break
            else:
                seed_list = flag2
                candidate = getevalcanidate(seed_list, G,K)
        cands.append(seed_list)
        print(cands)
        maxcand = cands[0]
        maxmcand = computeM(cands[0],G)
        for cand in cands:
            candm = computeM(cand,G)
            if(candm>maxmcand):
                maxcand = cand
                maxmcand = candm
        result.append(maxcand)
        i = i+1
    #保存结果
    print("result:"+str(result))
    max = result[0]
    maxm = computeM(max,G)
    for community in result:
        m = computeM(community,G)
        print(str(community) + "的M" + str(m))
        if(m>maxm):
            maxm = m
            max = community
    print(max)
    return max



def gptselectnodewithoutns(seed,G,iteration,K,SK,promptselect):#没有节点补充的算法，详细注释参考有节点补充的算法.
    """
    使用不具有节点补充的算法，输入种子节点、图结构、迭代次数、参数K、是否使用补充知识、提示符，获得最终社区。通过多次迭代，使用GPT进行节点选择，获得最终社区。

    参数:
    seed (int): 初始种子节点。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    iteration (int): 迭代次数。
    K (int): 潜在节点的数量。
    SK (bool): 是否包含补充知识的标志。
    promptselect (int): 提示符选择符，确定使用哪种提示策略。

    返回值:
    list: 最终的社区节点列表。
    """
    i=0
    result = []
    while(i<iteration):
        seed_list = [seed]
        seed_list11 = []
        candidate = getevalcanidate(seed_list, G,K)  #寻找潜在节点
        count = 0
        count2 = 0
        sleepcount = 0
        repeat = len(seed_list) - len(set(seed_list))
        k = len(seed_list) / 3
        stop = 0
        while ((len(candidate) > 0) and (repeat<k)): #判断算法是否终止
            if len(candidate) == 1:
                seed_list.append(candidate[0])
                candidate = getevalcanidate(seed_list, G,K)
                repeat = len(seed_list) - len(set(seed_list))
                k = len(seed_list) / 3
                continue
            seed_list = llms(seed_list, G, 1,K,SK,promptselect) # 节点选择
            repeat = len(seed_list) - len(set(seed_list))
            k = len(seed_list) / 3
            candidate = getevalcanidate(seed_list, G,K)
            sleepcount = sleepcount + 1
            if (sleepcount % 10 == 0)::#缓解超过访问限制
                time.sleep(10)
        result.append(seed_list)
        i = i+1
    # 保存结果
    print("result:"+str(result))
    max = result[0]
    maxm = computeM(max, G)
    for community in result:
        m = computeM(community, G)
        print(str(community) + "的M" + str(m))
        if (m > maxm):
            maxm = m
            max = community
    print("返回的社区是",max)
    return max


def gpt_communityexpansion(seed,G,ns,iteration,K,SK,promptselect):
    """
    根据ns判断是否进行节点补充，并选择不同的算法进行社区扩展。

    参数:
    seed (int): 初始种子节点。
    G (dict): 图结构，使用字典表示，其中键是节点，值是与该节点相邻的节点列表。
    ns (bool): 判断是否进行节点补充的标志。
    iteration (int): 迭代次数，决定社区扩展的循环次数。
    K (int): 潜在节点的数量。
    SK (bool): 是否包含补充知识的标志。
    promptselect (int): 提示符选择符，确定使用哪种提示策略。

    返回值:
    list: 最终的社区节点列表。
    """
    if(iteration<1):
        print("循环次数不能小于1")
        return null
    if(ns):
        return gptselectnodewithns(seed,G,iteration,K,SK,promptselect)
    else:
        return gptselectnodewithoutns(seed,G,iteration,K,SK,promptselect)


#计算当前社区和真实社区的评价指标
def eval_scores(pred_comm: Union[List, Set],
                 true_comm: Union[List, Set]) -> (float, float, float, float):
    """
    计算当前社区与真实社区之间的评价指标：精确率 (Precision), 召回率 (Recall), F1 值 (F1 Score) 和 Jaccard 相似度 (Jaccard Similarity)。

    参数:
    pred_comm (Union[List, Set]): 预测的社区节点列表或集合。
    true_comm (Union[List, Set]): 真实的社区节点列表或集合。

    返回值:
    tuple: 包含四个浮点数，分别表示精确率、召回率、F1 值和 Jaccard 相似度，结果保留四位小数。
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return round(p, 4), round(r, 4), round(f, 4), round(j, 4)



