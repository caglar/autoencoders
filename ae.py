import theano
import theano.tensor as T
from layer import AEHiddenLayer
import numpy

class CostType:
    MeanSquared = "MeanSquaredCost"
    CrossEntropy = "CrossEntropy"

class Autoencoder(object):

    def __init__(self,
            input,
            nvis,
            nhid,
            rnd=None,
            bhid=None,
            cost_type=CostType.CrossEntropy,
            L2_reg=-1,
            L1_reg=-1,
            bvis=None):

        self.input = input
        self.nvis = nvis
        self.nhid = nhid
        self.bhid = bhid
        self.bvis = bvis

        if cost_type == CostType.MeanSquared:
            self.cost_type = CostType.MeanSquared
        elif cost_type == CostType.CrossEntropy:
            self.cost_type = CostType.CrossEntropy

        if self.input is None:
            self.input = T.matrix('x')

        if rnd is None:
            self.rnd = numpy.random.RandomState(1231)
        else:
            self.rnd = rnd

        self.hidden = AEHiddenLayer(input, nvis, nhid, rng=rnd)
        self.params = self.hidden.params

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.L1 = 0
        self.L2 = 0

        if L1_reg != -1:
            self.L1 += abs(self.hidden.W).sum()

        if L2_reg != -1:
            self.L2 += (self.hidden.W**2).sum()

        if input is not None:
            self.x = input
        else:
            self.x = T.dmatrix('x_input')

    def encode(self, x_in=None):
        if x_in is None:
            x_in = self.x
        return T.nnet.sigmoid(T.dot(x_in, self.hidden.W) + self.hidden.b)

    def encode_linear(self, x_in=None):
        if x_in is None:
            x_in = self.x
        lin_output = T.dot(x_in, self.hidden.W) + self.hidden.b
        return T.nnet.sigmoid(lin_output), lin_output


    def decode(self, h):
        return T.nnet.sigmoid(T.dot(h, self.hidden.W_prime) + self.hidden.b_prime)

    def get_rec_cost(self, x_rec):
        """
        Returns the reconstruction cost.
        """
        if self.cost_type == CostType.MeanSquared:
            return T.mean(((self.x - x_rec)**2).sum(axis=1))
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean((T.nnet.binary_crossentropy(x_rec, self.x)).sum(axis=1))

    def get_sgd_updates(self, learning_rate, x_in=None):
        h = self.encode(x_in)
        x_rec = self.decode(h)

        cost = self.get_rec_cost(x_rec)
        if self.L1_reg != -1 and self.L1_reg != None:
            cost += self.L1_reg * self.L1

        if self.L2_reg != -1 and self.L2_reg != None:
            cost += self.L2_reg * self.L2

        gparams = T.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam
        return (cost, updates)

    def fit(self,
            data=None,
            learning_rate=0.1,
            batch_size=100,
            n_epochs=20,
            weights_file="out/ae_weights_mnist.npy"):
        """
        Fit the data to the autoencoder model. Basically this performs
        the learning.
        """
        if data is None:
            raise Exception("Data can't be empty.")

        index = T.lscalar('index')
        data_shared = theano.shared(numpy.asarray(data.tolist(), dtype=theano.config.floatX))
        n_batches = data.shape[0] / batch_size

        (cost, updates) = self.get_sgd_updates(learning_rate)

        train_ae = theano.function([index],
                                   cost,
                                   updates=updates,
                                   givens={
                                       self.x: data_shared[index * batch_size: (index + 1) * batch_size]
                                       }
                                   )

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
