import pandas as pd
import numpy as np
import copy
import networkx as nx
from networkx.algorithms.centrality import edge_betweenness_centrality
import operator
import matplotlib.pyplot as plt

def compute_jaccard_coef(H,S):
    Union_list = (set(H) | set(S))
    Intersection_list = (set(H) & set(S))
    return len(Intersection_list) / len(Union_list)

def compute_girvan_clusters(clusters,threshold=0.2):
    G = nx.Graph()
    G.add_nodes_from(range(len(clusters)))
    for i in range(len(clusters)-1):
        for j in range(i+1,len(clusters)):
            if(compute_jaccard_coef(clusters[i][0][2],clusters[j][0][2])>=threshold):
                G.add_edge(i,j)
    nx.draw(G,with_labels=True)
    plt.show()
    tot_clusters = len(list(nx.connected_components(G)))
    while(tot_clusters<9):
        centrality_measures = edge_betweenness_centrality(G)
        rem_edge = max(centrality_measures.items(),key=operator.itemgetter(1))[0]
        G.remove_edge(rem_edge[0],rem_edge[1])
        tot_clusters = len(list(nx.connected_components(G)))
    girvan_clusters = [list(c) for c in nx.connected_component_subgraphs(G)]
    nx.draw(G,with_labels=True)
    plt.show()
    return girvan_clusters

def main():
	data = pd.read_csv('AAAI.csv')
	clusters = []
	for i in range(len(data)):
	    point = [i,data.iloc[i]['Title'],data.iloc[i]['Topics'].split('\n'),data.iloc[i]['High-Level Keyword(s)']]
	    cluster = [point]
	    clusters.append(cluster)
	    Thresholds = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.25,0.3]
	for threshold in Thresholds:
		print('----------------------Threshold = '+str(threshold)+'-----------------')
		girvan_clusters_indices = compute_girvan_clusters(clusters,threshold)
		for i in range(len(girvan_clusters_indices)):
			print('Cluster '+str(i)+' size = '+str(len(girvan_clusters_indices[i])))
			print(girvan_clusters_indices[i])
		girvan_clusters = []
		for i in range(len(girvan_clusters_indices)):
			cluster = []
			for index in girvan_clusters_indices[i]:
				point = [index,data.iloc[index]['Title'],data.iloc[index]['Topics'].split('\n'),data.iloc[index]['High-Level Keyword(s)']]
				cluster.append(point)
			girvan_clusters.append(cluster)
		np.save('girvan_thresh'+str(threshold)+'.npy',girvan_clusters)

if __name__ == '__main__':
	main()