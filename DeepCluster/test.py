import faiss
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


dataset = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)

data = dataset[0]
data = data.astype(np.float32)
print(data.shape)

ncentroids = 4
niter = 20
verbose = True
d = data.shape[1]
index = faiss.IndexFlatL2(d)

kmeans = faiss.Clustering(d, ncentroids)
kmeans.verbose = verbose
kmeans.niter = niter
kmeans.train(data, index)
