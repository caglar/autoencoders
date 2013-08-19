from ca import ContractiveAutoencoder
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
    mu = numpy.mean(data, axis=1)
    sigma = numpy.std(data, axis=1)

    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("Std dev should not be zero")

    norm_data = (data - mu) / sigma
    return norm_data

if __name__=="__main__":
    fname = "/data/lisa/data/mnist/mnist.pkl"
    input = T.fmatrix("x_input")

    ds = Dataset()
    data = ds._get_data(fname)
    x_data = numpy.asarray(data[0][0][0:48000], dtype=theano.config.floatX)

    weights_file_l1 = "cae_mnist_weights_l1.npy"
    weights_file_l2 = "cae_mnist_weights_l2.npy"

    rnd = numpy.random.RandomState(1231)
    nhid_l1 = 900
    nhid_l2 = 900

    dae_l1 = ContractiveAutoencoder(input,
            nvis=28*28,
            nhid=nhid_l1,
            cost_type="CrossEntropy",
            rnd=rnd)

    #std_data = standardize(x_data)
    std_data = numpy.asarray(x_data, dtype=theano.config.floatX)

    dae_l1.fit(learning_rate=3*0.96*1e-2,
            data=std_data,
            weights_file=weights_file_l1,
            contraction_level=0.2,
            batch_size=128,
            n_epochs=60)

    dae_l1_obj_out = open("cae_l1_obj.pkl", "wb")
    pkl.dump(dae_l1, dae_l1_obj_out,  protocol=pkl.HIGHEST_PROTOCOL)


    dae_l1_out = dae_l1.encode(input)
    dae_l1_h = dae_l1.encode(std_data)
    dae_l1_h_fn = theano.function([], dae_l1_h)
    dae_l2_in = dae_l1_h_fn()
    dae_l2_in = numpy.asarray(dae_l2_in, dtype=theano.config.floatX)

    dae_l2 = ContractiveAutoencoder(dae_l1_out,
            nvis=nhid_l1,
            nhid=nhid_l2,
            cost_type="CrossEntropy",
            rnd=rnd)

    dae_l2.fit(learning_rate=0.95*1e-1,
            data=dae_l2_in,
            weights_file=weights_file_l2,
            contraction_level=0.25,
            batch_size=128,
            n_epochs=80)

    dae_l2_obj_out = open("cae_l2_obj.pkl", "wb")
    pkl.dump(dae_l2, dae_l2_obj_out,  protocol=pkl.HIGHEST_PROTOCOL)


    samples = dae_l2.sample(dae_l2_in[10], 40)
    numpy.save(open("data_samples_l2_dae.npy", "wb"), samples)
