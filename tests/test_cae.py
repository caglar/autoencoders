from ca import ContractiveAutoencoder
from dataset import Dataset
import theano.tensor as T
import numpy

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist_all.pickle"
    print "Started training CA on data %s " % (fname)
    ds = Dataset()
    ds.setup_dataset(data_path=fname, train_split_scale=0.8)
    x_data = ds.Xtrain
    input = T.dmatrix("x_input")
    weights_file = "../out/cae_mnist_weights.npy"

    rnd = numpy.random.RandomState(1231)
    dae = ContractiveAutoencoder(input, nvis=28*28, nhid=600, rnd=rnd)
    dae.fit(data=x_data, weights_file=weights_file, n_epochs=100)
