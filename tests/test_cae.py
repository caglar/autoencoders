from ca import ContractiveAutoencoder
from dataset import Dataset
import theano
import theano.tensor as T
import numpy

import cPickle as  pkl
theano.subtensor_merge_bug=False

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist_all.pickle"
    print "Started training CA on data %s " % (fname)
    ds = Dataset()
    data = ds._get_data(fname)
    x_data = numpy.asarray(data[0][0:42000], dtype=theano.config.floatX)
    input = T.fmatrix("x_input")
    weights_file = "../out/cae_mnist_weights.npy"

    rnd = numpy.random.RandomState(1231)
    powerup = PowerupAutoencoder(input, nvis=28*28, nhid=512, num_pieces=10, rnd=rnd)

    powerup.fit(data=x_data,
            weights_file=weights_file,
            shuffle_data=True,
            learning_rate=0.08,
            contraction_level=1,
            n_epochs=140)

    pae_obj_out = open("pae_data.pkl", "wb")
    pkl.dump(cae, cae_obj_out)

    single_data = numpy.array(x_data[0]).reshape((1, 28*28))

#    samples = []
#    for i in xrange(100):
#        samples.append(cae.sample(single_data, 0.1, 120))
#    numpy.save("data_samples.npy", samples)
