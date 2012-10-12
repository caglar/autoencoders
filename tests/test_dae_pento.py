from da import DenoisingAutoencoder
from dataset import Dataset
import theano.tensor as T
import numpy

if __name__=="__main__":
    fname = "/data/lisa/data/pentomino/pento64x64_40k_seed_5365102867_64patches.npy"
    ds = Dataset()
    ds.setup_dataset(data_path=fname, train_split_scale=0.4)
    x_data = ds.Xtrain
    input = T.dmatrix("x_input")
    rnd = numpy.random.RandomState(1231)
    dae = DenoisingAutoencoder(input,
            nvis=64*64,
            nhid=1500,
            rnd=rnd)

    dae.fit(data=x_data,
            learning_rate=0.04,
            n_epochs=32,
            weights_file="out/dae_weights_pento.npy")
