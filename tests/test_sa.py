from sa import SparseAutoencoder
from dataset import Dataset
import theano.tensor as T
import numpy

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist_all.pickle"
    print "Started training SA on data %s " % (fname)
    ds = Dataset()
    ds.setup_dataset(data_path=fname, train_split_scale=0.8)
    x_data = ds.Xtrain
    input = T.dmatrix("x_input")
    rnd = numpy.random.RandomState(1231)
    dae = SparseAutoencoder(input, nvis=28*28, nhid=2000, rnd=rnd)
    dae.fit(data=x_data)
