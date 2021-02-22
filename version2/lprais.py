
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

#  The similarity scores of the target layers are calculated by using the similarity scores of each layer.   
def rais_function(level_sim_dic, alpha):                          

    sim_dic = {}                                                                     
    for jdd4 in  level_sim_dic[alpha].keys():        
        sim_dic[jdd4] = 0

    for level in level_sim_dic.keys():
        if level == alpha:                                              
            maxvalue0 = max(level_sim_dic[level].values())
            minvalue0 = min(level_sim_dic[level].values())
            for jdd,simv in  level_sim_dic[level].items():
                level_sim_dic[level][jdd] = (simv -  minvalue0) / ( maxvalue0 - minvalue0)
                
            maxvalue1 = 1
            minvalue1 = 0
            sigma2 = ((maxvalue1 - minvalue1)**2) / 4
            for jdd2,simv2 in  level_sim_dic[level].items():                            
                if sigma2 == 0:
                    level_sim_dic[level][jdd2] = 1
                else:
                    kore1 = math.exp( - ((simv2 - maxvalue1)**2) / (2 * sigma2))
                    kore2 = math.exp( - ((simv2 - minvalue1)**2) / (2 * sigma2))
                    level_sim_dic[level][jdd2] =  kore1 / (kore1 + kore2)
                sim_dic[jdd2] += level_sim_dic[level][jdd2]    
        else:
            for jdd3 in  level_sim_dic[level].keys():
                if level_sim_dic[level][jdd3] == 1:
                    level_sim_dic[level][jdd3] = math.exp(2) / (1 + math.exp(2))
                else:
                    level_sim_dic[level][jdd3] = 1 / (1 + math.exp(2))

                sim_dic[jdd3] += level_sim_dic[level][jdd3]  
            
    return sim_dic    
      
          
def lprais_function(train_graph, sim_method, alpha, graph):                  
   
    ln = layerNum(graph)  
    # Computing the similarity scores of all unknown edges of the target layer in a multiplex network.                                       
    sim_dic_target = similarities(train_graph, sim_method)
    non_edge_list1 = [pair(u, v) for u, v in nx.non_edges(train_graph)]
    for key in non_edge_list1:
        sim_dic_target[key] = sim_dic_target[key] if key in sim_dic_target.keys() else 0
    
    level_sim_dic = {}                         # Storing the similarity scores of node pairs in each layer.        
    for i in range(ln):
        if i + 1 == alpha:
            level_sim_dic[i+1] = sim_dic_target                    
        else:                                          
            G = nx.read_edgelist(graph + str(i + 1) + '.edgelist', nodetype = int)
            sim_dic_pre = {}
            for key2 in non_edge_list1:
                sim_dic_pre[key2] = 1 if key2 in G.edges() else 0
            level_sim_dic[i+1] = sim_dic_pre
                      
    sim_dict = rais_function(level_sim_dic, alpha)
        
    return sim_dict    


















