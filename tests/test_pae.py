from pae import PowerupAutoencoder
from dataset import Dataset
import theano
import theano.tensor as T
import numpy

import cPickle as  pkl
theano.subtensor_merge_bug=False

if __name__ == "__main__":

    fname = "/data/lisa/data/mnist/mnist.pkl"
    print "Started training PAE on data %s " % (fname)
    ds = Dataset()
    data = ds._get_data(fname)
    x_data = numpy.asarray(data[0][0], dtype=theano.config.floatX)
    input = T.fmatrix("x_input")
    weights_file = "../out/pae_mnist_weights.npy"

    rnd = numpy.random.RandomState(1231)

    powerup = PowerupAutoencoder(input,
                                 nvis=28*28,
                                 nhid=300,
                                 momentum=0.54,
                                 rho=0.94,
                                 num_pieces=10,
                                 #max_col_norm=1.9368,
                                 cost_type="MeanSquaredCost",
                                 L2_reg=1.2*1e-5,
                                 L1_reg=6.62*1e-6,
                                 L1_act_reg=1.8*1e-4,
                                 #p_decay=5.4*1e-4,
                                 tied_weights=False,
                                 rnd=rnd)
    lr_scalers = {}
    lr_scalers["W"] = 0.09
    lr_scalers["b"] = 0.09
    lr_scalers["power"] = 0.01
    bs=120

    powerup.fit(data=x_data,
                weights_file=weights_file,
                batch_size=bs,
                lr_scalers=lr_scalers,
                corruption_level=0.01,
                shuffle_data=True,
                learning_rate=0.54*1e-3,
                n_epochs=320)

    pae_obj_out = open("pae_data.pkl", "wb")
    pkl.dump(powerup, pae_obj_out)
    encoded = powerup.encode(input, bs)
    decoded = powerup.decode(encoded)
    rec_fn = theano.function([input], decoded)
    reconstructed = rec_fn(data[0][0][0:bs])
    numpy.save("reconstructions.npy", reconstructed)
