         
import lp
import os
t = 20                                                      # training times
p = 0.1                                                     # proportion of the observed links as the testing set
suf = str(p * 10)
 
def layerNum(file):                                         #   Calculating the number of layers in a multiplex network.
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


path = './DataSet/'

networks = [                                                # All the multiplex networks used in the experiment.
    'Vickers/Vicker',  #0
    'CKM/CKM',  #1
    'Lazega/Lazega',  #2
    'Aarhus/Aarhus',  #3
    'Kapferer/Kapferer',  #4
    'Krackhardt/Krackhardt',  #5
    'Celegans/celegans',  #6
    'TF/TF',  #7
]

results = [
    './results/Vicker_',       './results/CKM_',           './results/lazega_',     './results/Aarhus_', 
    './results/Kapferer_',     './results/Krackhardt_',    './results/Celegans_',   './results/TF_'
]

net_ids = [0,1,2,3,4,5,6,7]                                # Selecting the multiplex networks used in the experiment.                         

graph_file_list = []
result_file_list = []

for i in net_ids:
    graph_file_list.append(path + networks[i])
    result_file_list.append(results[i] + suf)

sim_methods = [                                           # The basic similarity indexes.
    'CN',        # 0 
    'RA',        # 1
    'Jaccard',   # 2
    'PA'         # 3
]

method_ids = [0,1,2,3]                                    # Select the basic similarity indexes.           
sim_method_list = [sim_methods[i] for i in method_ids]
sigma = 'HALFRANGE'                                      # In the experiment of Section 4.2, sigma can be set to HALFRANGE, MIDRANGE, SD, MEAN or MEDIAN.

for i in range(len(graph_file_list)):
    graph_file = graph_file_list[i]
    result_file = result_file_list[i]
    print(graph_file)                                        
    for alpha in range(1,layerNum(graph_file)+1):
        out_file = open(result_file+'_lprais_'+str(alpha), 'w')  # Opening the result file.
        out_file.write( 'Relevance\tMethod\tAUC\tRanking_Score\ttime (us)\tPrecision (10)\n')           
        Rel = ['CSL','PCC','AASN']                     # layer similarity measures
        Phi = [ 0.5]                                   # 0： only target layer is considered；1：only auxiliary layers are considered；0.5：both target layer and auxiliary layers are considered.
        for method in sim_method_list:
            for rele in Rel:
                for phi in Phi:
                    print(method)
                    out_file.write(rele + '\t' + method + '\t')
                    lp.LP(graph_file, out_file, method, t, p, alpha, rele, phi, sigma)          
                    out_file.flush()                       
                out_file.write('\n')
        out_file.close()
    
            
