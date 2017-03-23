
import pickle
from sklearn.cluster import KMeans
import numpy as np
import glob
import os

class Node():
	def __init__(self, value,d,k):
		# self.data = [ data[i] for i in range(d) ]
		self.data=np.ones(d)
		self.data.dtype = "float16"
		self.data = value 
		self.children = [ None for i in range(k) ]
	
class Leaf():
	def __init__(self, value):
		self.data=np.ones(d)
		self.data = value 
		self.weight =  np.array( [0 for i in xrange(1,6049)], dtype='uint16')
		
class Tree:
	def __init__(self):
		self.root = None
		self.size =0 

	def insert(self, data, curr,i):
		n = Node(data,d,k)
		if self.root == None :
			self.root = n
		else :
			curr.children[i] = n
		return n	
	
	def insert_leaf(self, data, curr,i):
		n = Leaf(data)
		if self.root == None :
			self.root = n
		else :
			curr.children[i] = n
		return n	

 
	def inorder(self, n):
		if self.root == None :
			return None
		elif n == None :
			return None	
		else :
			for i in range(k):
				self.inorder(n.children[i])		
			self.print_value(n)
			
	def matching(self,X,n):
		if n == None: 
			return None
		else :
			n.frequency = n.frequency + 1 
			temp = sys.maxint  
			mini = sys.maxint   
			pos = None
			for i in range(k):
				temp = np.linalg.norm(X- n.children[i].data)
				if temp == None:
					continue 
			 	elif temp < mini:
			 		mini = temp 
			 		pos = n.children[i]
			if pos == None   :
				return n 		
			return self.matching(X,pos) 			

	def check(self, n):
		if self.root == None :
			return None
		elif n == None :
			return None	
		else :
			for i in range(k):
				self.check(n.children[i])		
			print n.data
	
	def maxlayer(self, curr):
		if hasattr(curr, 'children'):
			return 1 + self.maxlayer( curr.children[0] )
		else :  
			return 1 
	
def assign(X,curr,l):
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	for j in range(k):
		if l != l_max:
			T1.insert(kmeans.cluster_centers_[j],curr,j)	
			X_temp = X[kmeans.labels_[:] == j]
			assign(X_temp,curr.children[j],l+1)
		else:
			T1.insert_leaf(kmeans.cluster_centers_[j],curr,j)	
	return 0
		
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


## assignment 1 
#---------------#---------------#---------------#---------------#---------------#---------------
global d
d=64
global k
k=10
global i
i=0
global l_max
l_max=4;
global T1
T1 = Tree();

def setup():
	
	
	filename ='3m_high_tack_spray_adhesive.dat'
	mypoints=np.genfromtxt(filename, delimiter=',');
	count = 0 
	for filename in glob.glob(os.path.join("/home/pathak/vision/assign 1/done_points/", '*.dat')):
		X_temp=np.genfromtxt(filename, delimiter=',');
		mypoints = np.concatenate((mypoints, X_temp), axis=0);
		count = count + 1 
	print len(mypoints) ,count 
	kmeans = KMeans(n_clusters=1, random_state=0).fit(mypoints);	
	T1.insert(kmeans.cluster_centers_[0],None,0);
	assign(mypoints,T1.root,2);
	save_object(T1, 'T_final.pkl')
	np.save('All_data_points', mypoints)
	print T1.maxlayer(T1.root)
	#---------------#---------------#---------------#---------------#---------------#---------------

# setup()

