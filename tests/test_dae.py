from da import DenoisingAutoencoder
from dataset import Dataset
import theano.tensor as T
import numpy

if __name__=="__main__":
    #fname = "/data/lisa/data/mnist/mnist_all.pickle"
    fname = "/data/lisa/data/pentomino/"
    ds = Dataset()
    ds.setup_dataset(data_path=fname, train_split_scale=0.8)
    x_data = ds.Xtrain
    input = T.dmatrix("x_input")
    rnd = numpy.random.RandomState(1231)
    dae = DenoisingAutoencoder(input, nvis=28*28, nhid=500, rnd=rnd)
    dae.fit(data=x_data)
