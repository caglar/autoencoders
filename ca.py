import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy

from ae import Autoencoder, CostType

#Contractive Autoencoder implementation.
class ContractiveAutoencoder(Autoencoder):

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
        super(ContractionAutoencoder, self).__init__(input, nvis, nhid, rnd, bhid, cost_type, bvis)
        if not theano_rng :
            theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.theano_rng = theano_rng

    def contraction_penalty(self, contraction_level=0.0):

    def fit(self,
            data=None,
            learning_rate=0.1,
            batch_size=100,
            n_epochs=60,
            corruption_level=0.5,
            weights_file="out/dae_weights_mnist.npy"):

        if data is None:
            raise Exception("Data can't be empty.")

        index = T.lscalar('index')
        data_shared = theano.shared(numpy.asarray(data.tolist(), dtype=theano.config.floatX))
        n_batches = data.shape[0] / batch_size
        corrupted_input = self.corrupt_input(data_shared, corruption_level)
        (cost, updates) = self.get_sgd_updates(learning_rate, corrupted_input)

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
