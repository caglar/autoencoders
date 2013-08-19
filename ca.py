import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy

from ae import Autoencoder, CostType, Nonlinearity
from collections import OrderedDict

#Contractive Autoencoder implementation.
class ContractiveAutoencoder(Autoencoder):

    def __init__(self,
            input,
            nvis,
            nhid,
            rnd=None,
            theano_rng=None,
            bhid=None,
            sigma=0.06,
            nonlinearity=Nonlinearity.SIGMOID,
            cost_type=CostType.MeanSquared,
            bvis=None):
        self.sigma = sigma
        # create a Theano random generator that gives symbolic random values
        super(ContractiveAutoencoder, self).__init__(input, nvis, nhid, rnd, bhid, cost_type,
                nonlinearity=nonlinearity, sparse_initialize=True, bvis=bvis)
        if not theano_rng :
            theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.theano_rng = theano_rng

    def get_linear_hidden_outs(self, x_in=None):
        if x_in is None:
            x_in = self.x
        return T.dot(x_in, self.hidden.W) + self.hidden.b

    def contraction_penalty(self, h, linear_hid, contraction_level=0.0, batch_size=-1):
        """
        Compute the contraction penalty in the way that Ian describes in his e-mail:
        https://groups.google.com/d/topic/pylearn-dev/iY7swxgn-xI/discussion
        """
        if batch_size == -1 or batch_size == 0:
            raise Exception("Invalid batch_size!")

        grad = T.grad(h.sum(), linear_hid)
        jacob = T.dot(T.sqr(grad), T.sqr(self.hidden.W.sum(axis=0)))
        frob_norm_jacob = T.sum(jacob) / batch_size
        contract_pen = contraction_level * frob_norm_jacob
        return contract_pen

    def get_ca_sgd_updates(self, learning_rate, contraction_level, batch_size, x_in=None):
        h, linear_hid = self.encode_linear(x_in)
        x_rec = self.decode(h)
        cost = self.get_rec_cost(x_rec)
        contract_penal = self.contraction_penalty(h, linear_hid, contraction_level, batch_size)
        cost = cost + contract_penal
        gparams = T.grad(cost, self.params)
        updates = OrderedDict({})
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam
        return (cost, updates)

    def sample(self, x, K):

        if x.ndim == 1:
            x = x.reshape(1, x.shape[0])
        hn = self.encode(x)
        W = self.params[0]
        ww = T.dot(W.T, W)
        samples = []
        for _ in range(K):
            s = hn * (1. - hn)
            jj = ww * s.dimshuffle(0, 'x', 1) * s.dimshuffle(0, 1, 'x')
            alpha = self.srng.normal(size=hn.shape,
                    avg=0.,
                    std=self.sigma,
                    dtype=theano.config.floatX)

            delta = (alpha.dimshuffle(0, 1, 'x') * jj).sum(1)

            zn = self.decode(hn + delta)
            hn = self.encode(zn)
            #zn2 = self.decode(hn)
            samples.append(zn.eval())
        return samples

    def fit(self,
            data=None,
            learning_rate=0.1,
            batch_size=100,
            n_epochs=22,
            contraction_level=0.1,
            shuffle_data=True,
            weights_file="out/cae_weights_mnist.npy"):

        if data is None:
            raise Exception("Data can't be empty.")

        index = T.iscalar('index')
        data = numpy.asarray(data.tolist(), dtype="float32")
        data_shared = theano.shared(data)
        n_batches = data.shape[0] / batch_size
        (cost, updates) = self.get_ca_sgd_updates(learning_rate, contraction_level, batch_size)

        train_ae = theano.function([index],
                                   cost,
                                   updates=updates,
                                   givens={self.x: data_shared[index * batch_size: (index + 1) * batch_size]})

        print "Started the training."
        ae_costs = []
        for epoch in xrange(n_epochs):
            if shuffle_data:
                print "shuffling the dataset"
                numpy.random.shuffle(data)
                data_shared.set_value(data)
            print "Training at epoch %d" % epoch
            for batch_index in xrange(n_batches):
                ae_costs.append(train_ae(batch_index))
            print "Training at epoch %d, %f" % (epoch, numpy.mean(ae_costs))

        print "Saving files..."
        numpy.save(weights_file, self.params[0].get_value())
        return ae_costs
