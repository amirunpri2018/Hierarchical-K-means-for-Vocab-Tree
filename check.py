from karytree import *

import pickle
import numpy as np
import glob
import os

from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity , pairwise_distances
from scipy.sparse import csr_matrix
from numpy import linalg as LA

import time
import psutil
global str
from scipy.spatial import distance

global tmp_Match

level_max=4
k=10
No_of_leaf  = 1000 ; 

tf_idf_table =  np.array( [[0 for i in xrange(1,6049)] for  j in range(No_of_leaf)], dtype='float16') 
# 1000 X 6048
Freq_test =  np.array( [0 for i in range(No_of_leaf) ], dtype='int16') 
# 1 X 1000
IDF = np.array( [0 for i in range(No_of_leaf) ], dtype='float16')
# 1 X 1000 
Matching  = np.array( [0 for i in range(6048) ], dtype='float16')
# 1 X 6048

cordword_table =  np.array( [[0 for i in xrange(1000)] for  j in range(6048)], dtype='int16') 
# 1000 X 6048

def calc_left(level, pos): 
	if level == level_max: 
		return pos
	else : 
		temp =pos 
		return int(pos * pow(k,level_max -level))

def dist(A,B):
	temp = 0
	for i in range(d):
		temp = temp + (A[i] - B.data[i])*(A[i] - B.data[i])
	return float(temp)			 



def search(X,curr, level,left):
	if level == level_max:
		Freq_test[left] = Freq_test[left] + 1 
	else : 	
		pos = -1
		next_ = None 
		mini = 10000
		for j in range(k):
			temp = curr.children[j]
			distance = np.linalg.norm(temp.data-X)
			# if curr == T1.root : 
			# 	print distance
			if distance < mini :
				next_ = curr.children[j]
				mini = distance
				pos = j
		if next_ == None:
			return		 
		left_temp = calc_left(level +1 ,pos)		
		left = left + left_temp
		search(X, next_,level+1,left)
	


from PIL import Image
import subprocess



def print_images(index):
	filename = 'train_set/%04d.jpg'%(index)
	p = subprocess.Popen(["display", filename])
	time.sleep(2)
	p.kill()	


filenames = [i for i in os.listdir("/home/pathak/vision/assign 1/test/") if ".dat" in i ] 
filenames.sort()
for filename in filenames:
	print filename
	# if filename != 'campbells_chicken_noodle_soup.dat':
	# 	continue

	X_temp=np.genfromtxt("/home/pathak/vision/assign 1/test/"+filename, delimiter=',');
	with open('T_weighted_final.pkl', 'rb') as input:
		 T1 = pickle.load(input)	 

	# with open('Tfinal_weighted1.pkl', 'rb') as input:
	#  	T1 = pickle.load(input)
	with open('TF_IDF_transformer.pkl', 'rb') as input:
		TF_transformer = pickle.load(input)
	
	X_tf = np.load('TF_matrix.npy' )

	# print len(X_temp)

	for t in range(len(X_temp)):		
		X= X_temp[t]
		search(X,T1.root, 1,0)
		# print t
		# print X_temp[t][0]
		# raw_input("Start Next Round ...")
	
	Freq_test_tf = TF_transformer.transform(Freq_test) 
	Freq_test_tf = csr_matrix(Freq_test_tf, dtype='float32').toarray()
	tmp_Match = pairwise_distances(X_tf, Freq_test_tf, metric='euclidean', n_jobs=4)
	tmp_Match = np.reshape(tmp_Match, (6048,))
	

	# Z1 = distance.cdist(Freq_test_tf, X_tf, 'cosine')
	# Z2 = distance.cdist(Freq_test_tf, X_tf, 'minkowski', 1) 
	# Z3 = distance.cdist(Freq_test_tf, X_tf, 'minkowski', 2)
	Z3 = distance.cdist(Freq_test_tf, X_tf, 'minkowski', 1)
	# Z4 = distance.cdist(Freq_test_tf, X_tf, 'minkowski', 0.1)

	for i in Z3[0].argsort()[:10]: 
		print i + 1  
		print_images(i+1)		

	for i in range(No_of_leaf): 
		Freq_test[i] = 0  
	# break 	
	raw_input("Start Next Round ...")
