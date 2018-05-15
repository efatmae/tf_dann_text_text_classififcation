from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE

mnist = fetch_mldata("mnist-original")
batch = mnist.data[47000:47500]
data = batch.reshape(batch.shape[0], -1)
tsne = TSNE(n_components=2, random_state=0, init='pca', verbose=0)
tsne.fit_transform(data)
