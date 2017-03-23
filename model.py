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

def dist2(A,B):
	temp = 0 
	vec = B.data
	temp = np.dot(vec, A)
	temp2 = temp/( LA.norm(A) * LA.norm(vec) ) 
	return temp2   

def search(X,curr, level,left):
	if level == level_max:
		Freq_test[left] = Freq_test[left] + 1 
	else : 
		pos = -1
		next_ = None 
		mini = 1000000
		for j in range(k):
			temp = curr.children[j]
			distance = dist(X, temp)
			if distance < mini :
				next_ = curr.children[j]
				mini = distance
				pos = j 
		left_temp = calc_left(level +1 ,pos)		
		left = left + left_temp
		search(X, next_,level+1,left)
	
def search3(X,curr, level,left):
	if level == level_max:
		Freq_test[left] = Freq_test[left] + 1 
	else : 
		pos = -1
		next_ = None 
		mini = -1000000
		for j in range(k):
			temp = curr.children[j]
			distance = dist2(X, temp)
			if distance > mini :
				next_ = curr.children[j]
				mini = distance
				pos = j 
		left_temp = calc_left(level +1 ,pos)		
		left = left + left_temp
		search3(X, next_,level+1,left)
	

from PIL import Image
import subprocess




def print_images(index):
	filename = 'train_set/%04d.jpg'%(index-1)
	p = subprocess.Popen(["display", filename])
	time.sleep(1)
	p.kill()	

def main():
	filenames = [i for i in os.listdir("/home/pathak/vision/assign 1/test/") if ".dat" in i ] 
	filenames.sort()
	# filenames = glob.glob(os.path.join("/home/pathak/vision/assign 1/trail/", '*.dat'))
	for filename in filenames:
		print filename
		C = 0
		X_temp=np.genfromtxt("/home/pathak/vision/assign 1/test/"+filename, delimiter=',');
		
		for case in xrange(1,4):		
			with open('Tfinal_weighted%s.pkl'%(case), 'rb') as input:
			 	T1 = pickle.load(input)
			with open('TF_IDF_transformer%s.pkl'%(case), 'rb') as input:
				TF_transformer = pickle.load(input)
			X_tf = np.load('TF_matrix%s.npy'%(case) )
			
			for t in range(len(X_temp)):		
				X= X_temp[t]
				search3(X,T1.root, 1,0)

			# print len(X_temp)	,  sum(Freq_test)
			Freq_test_tf = TF_transformer.transform(Freq_test) 
			Freq_test_tf = csr_matrix(Freq_test_tf, dtype='float32').toarray()
			tmp_Match = pairwise_distances(X_tf, Freq_test_tf, metric='euclidean', n_jobs=4)
			tmp_Match = np.reshape(tmp_Match, (6048,))
			
			Z = distance.cdist(Freq_test_tf, X_tf, 'euclidean')
			print Freq_test_tf.shape , X_tf.shape , Z.shape
		# for i in Matching.argsort()[-10:][::-1]:
			
			# for i in tmp_Match.argsort()[-3:][::-1]:
			# 	print_images(i+1)			

			for i in tmp_Match.argsort()[:3]: 
				# print i 
				print_images(i+1)
			
			raw_input("Press Enter to continue...")

			break 
			for i in range(No_of_leaf): 
				Freq_test[i] = 0  
				

	# 	with open('Tfinal_weighted3.pkl', 'rb') as input:
	# 	 T1 = pickle.load(input)
	
	# with open('TF_IDF_transformer.pkl', 'rb') as input:
	# 	TF_transformer1 = pickle.load(input)

	# X_tf1 = np.load('TF_matrix.npy' )
	
	# C = 0

	


	# 	print len(Freq_test) , Freq_test , sum(Freq_test)
			
		
	
main()