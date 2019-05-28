import pandas as pd
import numpy as np
import copy

def compute_jaccard_coef(H,S):
    Union_list = (set(H) | set(S))
    Intersection_list = (set(H) & set(S))
    return len(Intersection_list) / len(Union_list)

def complete_linkage(temp_clusters):
    num_clusters = len(temp_clusters)
    while(num_clusters > 9):
        closest_val = 0
        closest_i = 0
        closest_j = 1
        for i in range(0,num_clusters):
            for j in range(i+1,num_clusters):
                curr_max = 1
                for k in range(len(temp_clusters[i])):
                    for l in range(len(temp_clusters[j])):
                        jacc_here = compute_jaccard_coef(temp_clusters[i][k][2],temp_clusters[j][l][2])
                        if(jacc_here < curr_max):
                            curr_max = jacc_here
                if(curr_max >= closest_val):
                    closest_val = curr_max
                    closest_i = i
                    closest_j = j
        temp_clusters[closest_i].extend(temp_clusters[closest_j])
        temp_clusters.pop(closest_j)
        num_clusters = len(temp_clusters)
    return temp_clusters

def single_linkage(temp_clusters):
    num_clusters = len(temp_clusters)
    while(num_clusters > 9):
        closest_val = 0
        closest_i = 0
        closest_j = 1
        for i in range(0,num_clusters-1):
            for j in range(i+1,num_clusters):
                curr_max = 0
                for k in range(len(temp_clusters[i])):
                    for l in range(len(temp_clusters[j])):
                        jacc_here = compute_jaccard_coef(temp_clusters[i][k][2],temp_clusters[j][l][2])
                        if(jacc_here > curr_max):
                            curr_max = jacc_here
                if(curr_max >= closest_val):
                    closest_val = curr_max
                    closest_i = i
                    closest_j = j
        temp_clusters[closest_i].extend(temp_clusters[closest_j])
        temp_clusters.pop(closest_j)
        num_clusters = len(temp_clusters)
    return temp_clusters

def main():
	data = pd.read_csv('AAAI.csv')
	clusters = []
	for i in range(len(data)):
	    point = [i,data.iloc[i]['Title'],data.iloc[i]['Topics'].split('\n'),data.iloc[i]['High-Level Keyword(s)']]
	    cluster = [point]
	    clusters.append(cluster)
	print("------------------Complete Linkage-----------------")
	init_clusters_complete = copy.deepcopy(clusters)
	complete_clusters = complete_linkage(init_clusters_complete)
	for i in range(len(complete_clusters)):
		print("cluster "+str(i)+' size = '+str(len(complete_clusters[i])))
		print(list(complete_clusters[i][j][0] for j in range(len(complete_clusters[i]))))
	np.save('complete.npy',complete_clusters)

	print("------------------Single Linkage-----------------")
	init_single_clusters = copy.deepcopy(clusters)
	single_clusters = single_linkage(init_single_clusters)

	for i in range(len(single_clusters)):
		print("cluster "+str(i)+' size = '+str(len(single_clusters[i])))
		print(list(single_clusters[i][j][0] for j in range(len(single_clusters[i]))))

	np.save('single.npy',single_clusters)
if __name__ == '__main__':
	main()