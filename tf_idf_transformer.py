from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics.pairwise import cosine_similarity , pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from scipy.spatial import distance

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


X= np.load('Freq_table.npy')

TF_transformer = TfidfTransformer(sublinear_tf=True)
TF_transformer.fit(X)
X_tf = TF_transformer.transform(X)

X_tf = csr_matrix(X_tf, dtype='float32').toarray()

X_heatmap = X_tf[:500, :]*100

Match = distance.cdist(X_heatmap, X_heatmap, 'minkowski', 0.5)
# Match = pairwise_distances(X_heatmap, X_heatmap, metric='euclidean', n_jobs=4)
# print("Match Made")
# sns.set()
# sns.heatmap(Match,  xticklabels=False, yticklabels=False)
# plt.show()
np.save('TF_matrix' , X_tf)
save_object(TF_transformer, 'TF_IDF_transformer.pkl')

