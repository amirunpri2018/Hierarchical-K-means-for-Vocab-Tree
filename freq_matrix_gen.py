from karytree import *
import pickle
import numpy as np
import glob
import os
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity , pairwise_distances

level_max=4
k=10
No_of_leaf  = 1000 ; 

Freq_test =  np.array( [0 for i in range(No_of_leaf) ], dtype='int16') 
# 1 X 1000
cordword_table =  np.array( [[0 for i in xrange(1000)] for  j in range(6048)], dtype='int16') 
# 6048 X 1000 

def check(curr, level ,check_sum):	
	C = 0 
	if level != level_max:
		for i in range(k):
			check_sum =  check(curr.children[i],level+1 , check_sum ) 	
		return check_sum 
	else:
		check_sum = check_sum + 1
		print check_sum, 
		print curr.weight.shape
		return check_sum

def frequencygen(curr, level ,check_sum):	
	C = 0 
	if level != level_max:
		for i in range(k):
			check_sum =  frequencygen(curr.children[i],level+1 , check_sum ) 	
		return check_sum 
	else:
		cordword_table[:,check_sum] = curr.weight
		check_sum = check_sum + 1
		return check_sum

def main () :
	with open('T_weighted_final.pkl', 'rb') as input:
		 T1 = pickle.load(input)	 
	frequencygen(T1.root, 1 ,0 )
	np.save('Freq_table', cordword_table)

main()	