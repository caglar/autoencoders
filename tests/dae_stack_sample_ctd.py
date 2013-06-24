from da import DenoisingAutoencoder
from dataset import Dataset
import theano.tensor as T
import theano
import numpy
import cPickle as  pkl

def standardize(data):
    """
    Normalize the data with respect to finding the mean and standard deviation of it
    and dividing by mean and standard deviation.
    """
    mu = numpy.mean(data, axis=0)
    sigma = numpy.std(data, axis=0)

    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("Std dev should not be zero")

    norm_data = (data - mu) / sigma
    return norm_data

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist_all.pickle"
    input = T.fmatrix("x_input")

    ds = Dataset()
    data = ds._get_data(fname)
    x_data = numpy.asarray(data[0][0:42000], dtype=theano.config.floatX)

    weights_file_l1 = "dae_mnist_weights_l1.npy"
    weights_file_l2 = "dae_mnist_weights_l2.npy"

    rnd = numpy.random.RandomState(1231)
    nhid_l1 = 800
    nhid_l2 = 800

    dae_l1 = DenoisingAutoencoder(input,
            nvis=28*28,
            nhid=nhid_l1,
            L1_reg=9.3*1e-5,
            L2_reg=8*1e-4,
            rnd=rnd)

    #std_data = standardize(x_data)
    std_data = numpy.asarray(x_data, dtype=theano.config.floatX)

    dae_l1.fit(learning_rate=9.96*1e-3,
            shuffle_data=True,
            data=std_data,
            weights_file=weights_file_l1,
            recons_img_file=None,
            corruption_level=0.096,
            batch_size=30,
            n_epochs=1400)

    dae_l1_obj_out = file("dae_l1_obj.pkl", "wb")
    pkl.dump(dae_l1, dae_l1_obj_out,  protocol=pkl.HIGHEST_PROTOCOL)

    dae_l1_out = dae_l1.encode(input)
    dae_l1_h = dae_l1.encode(std_data)
    dae_l1_h_fn = theano.function([], dae_l1_h)
    dae_l2_in = dae_l1_h_fn()
    dae_l2_in = numpy.asarray(dae_l2_in, dtype=theano.config.floatX)

    dae_l2 = DenoisingAutoencoder(dae_l1_out,
            L1_reg=1e-4,
            L2_reg=6*1e-4,
            nvis=nhid_l1,
            nhid=nhid_l2,
            rnd=rnd)

    dae_l2.fit(learning_rate=0.95*1e-2,
            data=dae_l2_in,
            shuffle_data=True,
            recons_img_file=None,
            weights_file=weights_file_l2,
            corruption_level=0.1,
            batch_size=25,
            n_epochs=1400)

    dae_l2_obj_out = file("dae_l2_obj.pkl", "wb")
    pkl.dump(dae_l2, dae_l2_obj_out,  protocol=pkl.HIGHEST_PROTOCOL)


    samples = dae_l2.sample(dae_l2_in[10], 0.1, 1000)
    numpy.save(open("data_samples_l2_dae.npy", "wb"), samples)

