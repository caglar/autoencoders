import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy

from ae import Autoencoder, CostType, Nonlinearity

class DenoisingAutoencoder(Autoencoder):

    def __init__(self,
            input,
            nvis,
            nhid,
            rnd=None,
            theano_rng=None,
            bhid=None,
            cost_type=CostType.MeanSquared,
            momentum=1,
            L1_reg=-1,
            L2_reg=-1,
            sparse_initialize=False,
            nonlinearity=Nonlinearity.TANH,
            bvis=None,
            tied_weights=True):

        # create a Theano random generator that gives symbolic random values
        super(DenoisingAutoencoder, self).__init__(input,
                nvis,
                nhid,
                rnd,
                bhid,
                cost_type,
                momentum,
                L1_reg=L1_reg,
                L2_reg=L2_reg,
                sparse_initialize=sparse_initialize,
                nonlinearity=nonlinearity,
                bvis=bvis,
                tied_weights=tied_weights)

        if not theano_rng :
            theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.theano_rng = theano_rng

    def corrupt_input(self, in_data, corruption_level):
        return self.theano_rng.binomial(self.x.shape, n=1, p=1-corruption_level,
                dtype=theano.config.floatX) * self.x

    def get_reconstructed_images(self, data):
        h = self.encode(x_in=data)
        x_rec = self.decode(h)
        return x_rec

    def debug_grads(self, data):
        gfn = theano.function([self.x], self.gparams[0])
        print "gradients:"
        print gfn(data)
        print "params:"
        print self.hidden.W.get_value()

    def fit(self,
            data=None,
            learning_rate=0.1,
            learning_rate_decay=None,
            batch_size=100,
            n_epochs=60,
            corruption_level=0.5,
            weights_file=None,
            sparsity_level=-1,
            sparse_reg=-1,
            shuffle_data=True,
            lr_scaler=1.0,
            recons_img_file="out/dae_reconstructed_pento.npy"):

        if data is None:
            raise Exception("Data can't be empty.")

        index = T.iscalar('index')
        data_shared = theano.shared(numpy.asarray(data.tolist(), dtype=theano.config.floatX))
        n_batches = data.shape[0] / batch_size

        corrupted_input = self.corrupt_input(data_shared, corruption_level)

        (cost, updates) = self.get_sgd_updates(learning_rate, lr_scaler=lr_scaler, batch_size=batch_size,
                sparsity_level=sparsity_level,
                sparse_reg=sparse_reg, x_in=corrupted_input)

        train_ae = theano.function([index],
                                   cost,
                                   updates=updates,
                                   givens={self.x: data_shared[index * batch_size: (index + 1) * batch_size]})

        print "Started the training."
        ae_costs = []
        batch_index = 0
        for epoch in xrange(n_epochs):
            idxs = numpy.arange(n_batches)
            numpy.random.shuffle(idxs)
            print "Training at epoch %d" % epoch
            for batch_index in idxs:
                ae_costs.append(train_ae(batch_index))
                if False:
                    print "Cost: ", ae_costs[-1]
                    self.debug_grads(data_shared.get_value()[batch_index * batch_size: (batch_index + 1) * batch_size])
            print "Training at epoch %d, %f" % (epoch, numpy.mean(ae_costs))

        if weights_file is not None:
            print "Saving weights..."
            numpy.save(weights_file, self.params[0].get_value())
        if recons_img_file is not None:
            print "Saving reconstructed images..."
            x_rec = self.get_reconstructed_images(data_shared)
            numpy.save(recons_img_file, x_rec)
        return ae_costs
