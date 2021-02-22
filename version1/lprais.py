
import networkx as nx
import numpy as np
import math
from LPMethod import similarities
import os
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
    
def layerNum(file):
    path = file[:file.rfind('/')]
    all_files = os.listdir(path)
    type_dict = dict()

    for f in all_files:
        if os.path.isdir(f):
            type_dict.setdefault('folder', 0)
            type_dict['folder'] += 1
        else:
            ext = os.path.splitext(f)[1]
            type_dict.setdefault(ext, 0)
            type_dict[ext] += 1
    return type_dict['.edgelist']


def pair(x, y):     
    if (x < y):
        return (x, y)
    else:
        return (y, x)
# The function of CSL
def CSL(a, b):
    
    nodes = []                                        
    for va in nx.nodes(a):
        if va not in nodes:
            nodes.append(va)
    for vb in nx.nodes(b):
        if vb not in nodes:
            nodes.append(vb)        
                    
    matrix1 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in a.edges():
                matrix1[i][j] = 1
                matrix1[j][i] = 1
    avector = matrix1.flatten().tolist()
    matrix2 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in b.edges():
                matrix2[i][j] = 1
                matrix2[j][i] = 1
    bvector = matrix2.flatten().tolist()

    return (1 - cosine(avector, bvector))

# The function of PCC
def PCC(a, b):
    
    nodes = []                                          
    for va in nx.nodes(a):
        if va not in nodes:
            nodes.append(va)
    for vb in nx.nodes(b):
        if vb not in nodes:
            nodes.append(vb)        
                    
    matrix1 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in a.edges():
                matrix1[i][j] = 1
                matrix1[j][i] = 1
    avector = matrix1.flatten().tolist()
    matrix2 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in b.edges():
                matrix2[i][j] = 1
                matrix2[j][i] = 1
    bvector = matrix2.flatten().tolist()

    return pearsonr(avector, bvector)[0]   


# The function of AASN
def AASN(a, b):
    
    nodes = []                                       
    for va in nx.nodes(a):
        if va not in nodes:
            nodes.append(va)
    for vb in nx.nodes(b):
        if vb not in nodes:
            nodes.append(vb)

    sim = np.zeros((2, len(nodes)))
    aa = [pair(u, v) for u, v in nx.edges(a)]
    bb = [pair(u, v) for u, v in nx.edges(b)]
    for i in nodes:
        for j in nodes:            
            if pair(i, j) in bb:
                sim[0, nodes.index(i)] += 1
            if ( (pair(i, j) in aa) and (pair(i, j) in bb) ):
                sim[1, nodes.index(i)] += 1
                    
    return sum(sim[1]) / sum(sim[0])
 
def relevance(r, a, b):
    
    if r == 'CSL':
        return CSL(a, b)
    if r == 'PCC':
        return PCC(a, b)    
    if r == 'AASN':
        return AASN(a, b)    
 
    
#  The similarity scores of the target layers are calculated by using the similarity scores of each layer and the interlayer correlation.   
def rais_function(level_sim_dic,level_relev_dic,alpha, phi, sigma):                          
    
    sum_rele = sum(level_relev_dic.values())        
    for key22,value22 in level_relev_dic.items():                                   
        level_relev_dic[key22] = value22 / sum_rele
    
    sim_dic = {}                                                                    
    for jdd4 in  level_sim_dic[alpha].keys():        
        sim_dic[jdd4] = 0

    for level in level_sim_dic.keys():                                              
        maxvalue0 = max(level_sim_dic[level].values())
        minvalue0 = min(level_sim_dic[level].values())
        for jdd,simv in  level_sim_dic[level].items():
            if maxvalue0 == minvalue0:
                level_sim_dic[level][jdd] = 0
            else:
                level_sim_dic[level][jdd] = (simv -  minvalue0) / ( maxvalue0 - minvalue0)
        list1 = list(level_sim_dic[level].values())                    
        maxvalue1 = max(level_sim_dic[level].values())
        minvalue1 = min(level_sim_dic[level].values())
        if sigma == 'HALFRANGE':
            sigma2 = ((maxvalue1 - minvalue1)**2) / 4
        if sigma == 'MEDIAN':
            sigma2 = (np.median(list1))**2
        if sigma == 'MIDRANGE':
            sigma2 = ((maxvalue1 + minvalue1)**2) / 4
        if sigma == 'SD':
            sigma2 = np.var(list1)
        if sigma == 'MEAN':
            sigma2 = (np.mean(list1))**2
            
        for jdd2,simv2 in  level_sim_dic[level].items():                             
            if sigma2 == 0 and sigma != 'MEDIAN':
                level_sim_dic[level][jdd2] = 1
            else:
                kore1 = math.exp( - ((simv2 - maxvalue1)**2) / (2 * sigma2))
                kore2 = math.exp( - ((simv2 - minvalue1)**2) / (2 * sigma2))
                level_sim_dic[level][jdd2] =  kore1 / (kore1 + kore2)

            if level == alpha:
                sim_dic[jdd2] += (level_sim_dic[level][jdd2] * level_relev_dic[level] * (1 - phi)) 
            else:
                sim_dic[jdd2] += (level_sim_dic[level][jdd2] * level_relev_dic[level] *  phi)          
            
    return sim_dic    

           
def lprais_function(train_graph, sim_method, alpha, graph, rele, phi, sigma):                  
   
    ln = layerNum(graph)  
    # Computing the similarity scores of all unknown edges of the target layer in a multiplex network.                                       
    sim_dic_target = similarities(train_graph, sim_method)
    non_edge_list1 = [pair(u, v) for u, v in nx.non_edges(train_graph)]
    for key in non_edge_list1:
        sim_dic_target[key] = sim_dic_target[key] if key in sim_dic_target.keys() else 0
    
    level_sim_dic = {}                         # Storing the similarity scores of node pairs in each layer.        
    level_relev_dic = {}                       # Storing the correlation between each layer and the target layer.
    for i in range(ln):
        if i + 1 == alpha:
            level_sim_dic[i+1] = sim_dic_target
            if rele == 'CSL' or rele == 'PCC' or rele == 'AASN':
                level_relev_dic[i+1] = 1 
            else:
                relev_dic_target = {}
                for key3 in non_edge_list1:
                    relev_dic_target[key3] = 1
                level_relev_dic[i+1] = relev_dic_target
                    
        else:                                          
            G = nx.read_edgelist(graph + str(i + 1) + '.edgelist', nodetype = int)
            sim_dic_pre = {}
            for key2 in non_edge_list1:
                sim_dic_pre[key2] = 1 if key2 in G.edges() else 0
            level_sim_dic[i+1] = sim_dic_pre
            level_relev_dic[i+1] = relevance(rele, train_graph, G)
                      
    sim_dict = rais_function(level_sim_dic,level_relev_dic,alpha, phi, sigma)
        
    return sim_dict    


















