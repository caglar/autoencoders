from ae import Autoencoder
from dataset import Dataset
import theano.tensor as T
import numpy

def test_ae():
    pass

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist_all.pickle"
    ds = Dataset()
    ds.setup_dataset(data_path=fname, train_split_scale=0.8)
    x_data = ds.Xtrain
    input = T.dmatrix("x_input")
    rnd = numpy.random.RandomState(1231)
    ae = Autoencoder(input, nvis=28*28, nhid=500, rnd=rnd)
    ae.fit(data=x_data)
