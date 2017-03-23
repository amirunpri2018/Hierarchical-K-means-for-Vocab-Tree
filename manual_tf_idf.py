import numpy as np
import pickle
from math import log 

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


X= np.load('Freq_table.npy')
X = np.asarray(X)
print X.shape

TF_IDF =  np.array([[0 for i in range(1000)] for i in range(6048)], dtype='float16')
print TF_IDF.shape

IDF =  np.array([0 for i in range(1000)], dtype='float16')

for i in range(6048): 
	# print X[i]	
	TF_IDF[i] = X[i].astype(float) / float(sum(X[i]))
	# print "----------"
	# print TF_IDF[i]


for i in range(1000):
	Y =6048 - len(X[X[:,i] == 0]) 	
	IDF[i] = log(6048/(1+Y)) 

FINAL  =TF_IDF * IDF

# 		TF_IDF[j,i] = ( X[j,i].astype(float)/sum(X[j,:]))*IDF 	


# np.save('TF_matrix_self' , X_tf)



