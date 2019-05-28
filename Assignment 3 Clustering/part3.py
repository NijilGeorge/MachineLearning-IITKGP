import pandas as pd
import numpy as np
from collections import Counter

def compute_mutual_info(clusters,label_counts,N=150):
	mutual_info = 0
	for i in range(len(clusters)):
		for label in label_counts:
			num_labels_in_cluster = 0
			for paper in clusters[i]:
				if(paper[3] == label):
					num_labels_in_cluster += 1
			if(num_labels_in_cluster >0):
				mutual_info += ((num_labels_in_cluster/N) * np.log2((N*num_labels_in_cluster)/(len(clusters[i])*label_counts[label])))
	return mutual_info

def compute_denom(clusters,label_counts,N=150):
	entropy_clusters = 0
	for i in range(len(clusters)):
		entropy_clusters -= ((len(clusters[i])/N)*np.log2(len(clusters[i])/N))
	entropy_class = 0
	for label in label_counts:
		entropy_class -= ((label_counts[label]/N)*np.log2(label_counts[label]/N))
	return (entropy_class+entropy_clusters)/2

def compute_nmi(clusters,label_counts,N=150):
	numerator = compute_mutual_info(clusters,label_counts,N)
	denominator = compute_denom(clusters,label_counts,N)
	return numerator/denominator

def main():
	data = pd.read_csv('AAAI.csv')
	label_counts={}
	for label in np.unique(data['High-Level Keyword(s)']):
		label_counts[label] = len(data[data['High-Level Keyword(s)'] == label])

	complete_clusters = np.load('complete.npy')
	print("Complete linkage clustering")
	print(compute_nmi(complete_clusters,label_counts))
	single_clusters = np.load('single.npy')
	print("Single linkage clustering")
	print(compute_nmi(single_clusters,label_counts))
	Thresholds = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.25]
	for threshold in Thresholds:
		girvan_clusters = np.load('girvan_thresh'+str(threshold)+'.npy')
		print("Girvan Newmann clustering with threshold "+str(threshold))
		print(compute_nmi(girvan_clusters,label_counts))

if __name__ == '__main__':
	main()