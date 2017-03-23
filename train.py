from karytree import *
import pickle
import numpy as np
import glob
import os
from numpy import linalg as LA

temp = 0
level_max=4

def check(curr, level):	
	if level != level_max:
		for i in range(k):
			check(curr.children[i],level+1) 	
	else:
		print curr.weight[0:3 ]
		return 0 

def dist(A,B):
	temp = 0 
	for i in range(d):
		temp = temp + (A[i] - B.data[i])*(A[i] - B.data[i])
	return float(temp)


def codeword(curr, level,index, X):
	if level == level_max:
		curr.weight[index] =curr.weight[index] + 1
	else : 
		next_ = None 
		mini = 1000000
		for j in range(k):
			temp = curr.children[j]
			distance = np.linalg.norm(temp.data-X)
			if distance < mini :
				next_ = curr.children[j]
				mini = distance
		return codeword(next_,level+1,index,X)	

def frequency_generator(T1): 
	temp_sum = 0
	for filename in glob.glob(os.path.join("/home/pathak/vision/assign 1/train_points/", '*.dat')):
		X_temp=np.genfromtxt(filename, delimiter=',');
		print filename
		index = (os.path.splitext(os.path.basename(filename ))[0]); 
		print index
		index = index.replace(".jpg", "")
		index = int(index) -1 
		temp_sum = temp_sum + 1
		print temp_sum
		for t in range(len(X_temp)):		
			X= X_temp[t]
			codeword(T1.root, 1,index, X)
		

def main():
	print d , k , level_max
	with open('T_final.pkl', 'rb') as input:
		T1 = pickle.load(input)
	# print T1.root.data
	frequency_generator(T1)
	save_object(T1, 'T_weighted_final.pkl')
	# print check(T1.root, 1)	
	
main()	