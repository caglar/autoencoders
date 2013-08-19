from pae import PowerupAutoencoder
from dataset import Dataset
import theano
import theano.tensor as T
import numpy

from pylearn2.datasets.preprocessing import Standardize, LeCunLCN, GlobalContrastNormalization
from pylearn2.datasets.tfd import TFD

import cPickle as  pkl
theano.subtensor_merge_bug=False

if __name__ == "__main__":
    weights_file = "../out/pae_mnist_enc_weights.npy"
    input = T.matrix("X", dtype=theano.config.floatX)
    tfd_ds = TFD("unlabeled")

    print "TFD shape: ", tfd_ds.X.shape
    gcn = GlobalContrastNormalization()
    standardizer = Standardize()
    lcn = LeCunLCN(img_shape=(48, 48), channels=[0])
    gcn.apply(tfd_ds, can_fit=True)
    standardizer.apply(tfd_ds, can_fit=True)
    lcn.apply(tfd_ds)

    rnd = numpy.random.RandomState(1231)

    powerup = PowerupAutoencoder(input,
                                 nvis=48*48,
                                 nhid=500,
                                 momentum=0.66,
                                 rho=0.92,
                                 num_pieces=4,
                                 cost_type="MeanSquaredCost",
                                 L2_reg=8.2*1e-5,
                                 L1_reg=1.2 * 1e-5,
                                 L1_act_reg=8.8*1e-4,
                                 tied_weights=False,
                                 rnd=rnd)

    lr_scalers = {}
    lr_scalers["W"] = 0.035
    lr_scalers["b"] = 0.05
    lr_scalers["power"] = 0.01
    bs=60

    powerup.fit(data=tfd_ds.X,
                weights_file=weights_file,
                batch_size=bs,
                lr_scalers=lr_scalers,
                corruption_level=0.02,
                shuffle_data=True,
                learning_rate=0.34*1e-3,
                n_epochs=120)

    pae_obj_out = open("pae_data.pkl", "wb")
    pkl.dump(powerup, pae_obj_out)
    encoded = powerup.encode(input, bs)
    decoded = powerup.decode(encoded)
    rec_fn = theano.function([input], decoded)
    reconstructed = rec_fn(tfd_ds.X[0:bs])

    numpy.save("reconstructions.npy", reconstructed)

