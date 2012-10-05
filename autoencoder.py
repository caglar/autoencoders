import theano
import theano.tensor as T

class Autoencoder(object):

    def __init__(self,
            input=None,
            nvis=None,
            nhid=None,
            batch_size=None,
            rnd=None,
            bhid=None,
            bvis=None):

        self.input = input
        self.nvis = nvis
        self.nhid = nhid
        self.batch_size = batch_size
        self.bhid = bhid
        self.bvis = bvis

    def encode(self, X):
        pass

    def decode(self, h):
        pass

    def fit(self, X, learning_rate, n_epochs):
        pass
