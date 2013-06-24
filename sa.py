import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy

from ae import Autoencoder, CostType

#Contractive Autoencoder implementation.
class SparseAutoencoder(Autoencoder):

    def __init__(self,
            input,
            nvis,
            nhid,
            rnd=None,
            theano_rng=None,
            bhid=None,
            cost_type=CostType.CrossEntropy,
            bvis=None):

        # create a Theano random generator that gives symbolic random values
        super(SparseAutoencoder, self).__init__(input, nvis, nhid, rnd, bhid, cost_type, bvis)
        if not theano_rng :
            theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.theano_rng = theano_rng

    def get_linear_hidden_outs(self, x_in=None):
        if x_in is None:
            x_in = self.x
        return T.dot(x_in, self.hidden.W) + self.hidden.b

    def kl_divergence(self, p, p_hat):
        term1 = p * T.log(p)
        term2 = p * T.log(p_hat)
        term3 = (1-p) * T.log(1 - p)
        term4 = (1-p) * T.log(1 - p_hat)
        return term1 - term2 + term3 - term4

    def sparsity_penalty(self, h, sparsity_level=0.05, sparse_reg=1e-3, batch_size=-1):
        if batch_size == -1 or batch_size == 0:
            raise Exception("Invalid batch_size!")
        sparsity_level = T.extra_ops.repeat(sparsity_level, self.nhid)
        sparsity_penalty = 0
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_level, avg_act)
        sparsity_penalty = sparse_reg * kl_div.sum()
        # Implement KL divergence here.
        return sparsity_penalty

    def get_sa_sgd_updates(self, learning_rate, sparsity_level, sparse_reg, batch_size, x_in=None):
        h = self.encode(x_in)
        x_rec = self.decode(h)
        cost = self.get_rec_cost(x_rec)
        sparsity_penal = self.sparsity_penalty(h, sparsity_level, sparse_reg, batch_size)
        cost = cost + sparsity_penal

        gparams = T.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam
        return (cost, updates)

    def fit(self,
            data=None,
            learning_rate=0.08,
            batch_size=100,
            n_epochs=22,
            sparsity_penalty=0.001,
            sparsity_level=0.05,
            weights_file="out/sa_weights_mnist.npy"):

        if data is None:
            raise Exception("Data can't be empty.")

        index = T.lscalar('index')
        data_shared = theano.shared(numpy.asarray(data.tolist(), dtype=theano.config.floatX))
        n_batches = data.shape[0] / batch_size
        (cost, updates) = self.get_sa_sgd_updates(learning_rate, sparsity_level, sparsity_penalty, batch_size)

        train_ae = theano.function([index],
                                   cost,
                                   updates=updates,
                                   givens={self.x: data_shared[index * batch_size: (index + 1) * batch_size]})

        print "Started the training."
        ae_costs = []
        for epoch in xrange(n_epochs):
            print "Training at epoch %d" % epoch
            for batch_index in xrange(n_batches):
                ae_costs.append(train_ae(batch_index))
            print "Training at epoch %d, %f" % (epoch, numpy.mean(ae_costs))

        print "Saving files..."
        numpy.save(weights_file, self.params[0].get_value())
        return ae_costs
