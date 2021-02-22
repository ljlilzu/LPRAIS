
import networkx as nx
import scipy as sp
import numpy as np
import math
import warnings


def similarities(graph, method, flag=False):

    if method == 'CN':
        return common_neighbors_index(graph, flag)
    if method == 'RA':
        return resource_allocation_index(graph, flag)
    if method == 'Jaccard':
        return jaccard_coefficient(graph, flag)
    if method == 'PA':
        return preferential_attachment_index(graph,flag)
    
    
def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
# CN
def common_neighbors_index(G, for_existing_edge=False):
    # for_existing_edge=True    calculating the common neighbors of the existing edges. 
    # for_existing_edge=False   calculating the common neighbors of the nonexistent edges. 
    
    sim_dict = {}    
    node_num = nx.number_of_nodes(G)
    for u in G.nodes():
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1
                        else:
                            sim_dict[(u,v)] = 1
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def

# PA
def preferential_attachment_index(G, for_existing_edge=False):
    # for_existing_edge=True    calculating the common neighbors of the existing edges. 
    # for_existing_edge=False   calculating the common neighbors of the nonexistent edges. 
    ebunch = nx.edges(G) if for_existing_edge else nx.non_edges(G)
    ebunch = [pair(u, v) for u, v in ebunch]                                     
    sim_dict = {}   
    degree_list = {}
    for v in G.nodes():
        degree_list[v] = nx.degree(G,v)

    for u, v in ebunch:
        s = degree_list[u] * degree_list[v]
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

# Jaccard
def jaccard_coefficient(G, for_existing_edge=False):
    # for_existing_edge=True    calculating the common neighbors of the existing edges. 
    # for_existing_edge=False   calculating the common neighbors of the nonexistent edges. 
        
    sim_dict = {}   
    node_num = nx.number_of_nodes(G)
    for u in G.nodes():
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1.0
                        else:
                            sim_dict[(u,v)] = 1.0
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    degree_list = {}
    for v in G.nodes():
        degree_list[v] = nx.degree(G,v)

    for (u, v) in sim_dict.keys():
        s = sim_dict[(u, v)]
        sim_dict[(u, v)] = s / (degree_list[u] + degree_list[v] - s)
    # end for
    
    return sim_dict
# end def


# RA
def resource_allocation_index(G, for_existing_edge=False):
    # for_existing_edge=True    calculating the common neighbors of the existing edges. 
    # for_existing_edge=False   calculating the common neighbors of the nonexistent edges. 

    sim_dict = {}     
    node_num = nx.number_of_nodes(G)
    degree_list = {}
    for v in G.nodes():
        degree_list[v] = nx.degree(G,v)

    for u in G.nodes():
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u, v) in sim_dict:
                            sim_dict[(u, v)] += 1 / degree_list[w]
                        else:
                            sim_dict[(u, v)] = 1 / degree_list[w]
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def

