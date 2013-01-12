from da import DenoisingAutoencoder
from dataset import Dataset
import theano.tensor as T
import numpy

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist_all.pickle"
    #fname = "/data/lisa/data/pentomino/"
    ds = Dataset()
    ds.setup_dataset(data_path=fname, train_split_scale=0.8)
    x_data = ds.Xtrain
    input = T.dmatrix("x_input")

    weights_file = "../out/dae_mnist_weights.npy"
    recons_file = "../out/dae_mnist_recons.npy"
    rnd = numpy.random.RandomState(1231)
    dae = DenoisingAutoencoder(input, nvis=28*28, nhid=600, rnd=rnd)
    dae.fit(learning_rate=0.1, data=x_data, weights_file=weights_file, n_epochs=100, recons_img_file=recons_file)
