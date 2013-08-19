import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layer import AEHiddenLayer
import numpy

from collections import OrderedDict

theano.config.warn.subtensor_merge_bug = False

class Nonlinearity:
    RELU = "rectifier"
    TANH = "tanh"
    SIGMOID = "sigmoid"

class CostType:
    MeanSquared = "MeanSquaredCost"
    CrossEntropy = "CrossEntropy"

class Autoencoder(object):

    def __init__(self,
            input,
            nvis,
            nhid=None,
            nvis_dec=None,
            nhid_dec=None,
            rnd=None,
            bhid=None,
            cost_type=CostType.MeanSquared,
            momentum=1,
            num_pieces=1,
            L2_reg=-1,
            L1_reg=-1,
            sparse_initialize=False,
            nonlinearity=Nonlinearity.TANH,
            bvis=None,
            tied_weights=True):

        self.input = input
        self.nvis = nvis
        self.nhid = nhid
        self.bhid = bhid
        self.bvis = bvis
        self.momentum = momentum
        self.nonlinearity = nonlinearity
        self.tied_weights = tied_weights
        self.gparams = None

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

        self.srng = RandomStreams(seed=1231)

        self.hidden = AEHiddenLayer(input,
                nvis,
                nhid,
                num_pieces=num_pieces,
                n_in_dec=nvis_dec,
                n_out_dec=nhid_dec,
                activation=None,
                tied_weights=tied_weights,
                sparse_initialize=sparse_initialize,
                rng=rnd)

        self.params = self.hidden.params

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.sparse_initialize = sparse_initialize

        self.L1 = 0
        self.L2 = 0

        if L1_reg != -1:
            self.L1 += abs(self.hidden.W).sum()
            if not tied_weights and 0:
                self.L1 += abs(self.hidden.W_prime).sum()

        if L2_reg != -1:
            self.L2 += (self.hidden.W**2).sum()
            if not tied_weights and 0:
                self.L2 += (self.hidden.W_prime**2).sum()

        if input is not None:
            self.x = input
        else:
            self.x = T.matrix('x_input', dtype=theano.config.floatX)

    def nonlinearity_fn(self, d_in=None, recons=False):
        if self.nonlinearity == Nonlinearity.SIGMOID:
            return T.nnet.sigmoid(d_in)
        elif self.nonlinearity == Nonlinearity.RELU and not recons:
            return T.maximum(d_in, 0)
        elif self.nonlinearity == Nonlinearity.RELU and recons:
            return T.nnet.softplus(d_in)
        elif self.nonlinearity == Nonlinearity.TANH:
            return T.tanh(d_in)

    def encode(self, x_in=None, center=True):
        if x_in is None:
            x_in = self.x
        act = self.nonlinearity_fn(T.dot(x_in, self.hidden.W) + self.hidden.b)
        if center:
            act = act - act.mean(0)
        return act

    def encode_linear(self, x_in=None):
        if x_in is None:
            x_in = self.x
        lin_output = T.dot(x_in, self.hidden.W) + self.hidden.b
        return self.nonlinearity_fn(lin_output), lin_output

    def decode(self, h):
        return self.nonlinearity_fn(T.dot(h, self.hidden.W_prime) + self.hidden.b_prime)

    def get_rec_cost(self, x_rec):
        """
        Returns the reconstruction cost.
        """
        if self.cost_type == CostType.MeanSquared:
            return T.mean(((self.x - x_rec)**2).sum(axis=1))
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean((T.nnet.binary_crossentropy(x_rec, self.x)).mean(axis=1))

    def kl_divergence(self, p, p_hat):
        return p * T.log(p) - T.log(p_hat) + (1 - p) * T.log(1 - p) - (1 - p) * T.log(1 - p_hat)

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

    def act_grads(self, inputs):
        h, acts = self.encode_linear(inputs)
        h_grad = T.grad(h.sum(), acts)
        return (h, h_grad)

    def jacobian_h_x(self, inputs):
        h, act_grad = self.act_grads(inputs)
        jacobian = self.hidden.W * act_grad.dimshuffle(0, 'x', 1)
        return (h, T.reshape(jacobian, newshape=(self.nhid, self.nvis)))

    def compute_jacobian_h_x(self, inputs):

        inputs = theano.shared(inputs.flatten())
        h = self.encode(inputs)
        #h = h.flatten()
        #inputs = inputs.flatten()
        #inputs = T.reshape(inputs, newshape=(self.nvis))
        J = theano.gradient.jacobian(h, inputs)
        return h, J

    def sample_one_step(self, x, sigma):
        #h, J_t = self.jacobian_h_x(x)
        h, J_t = self.compute_jacobian_h_x(x)
        eps = self.srng.normal(avg=0, size=(self.nhid, 1), std=sigma)
        jacob_w_eps = T.dot(J_t.T, eps)
        delta_h = T.dot(J_t, jacob_w_eps)
        perturbed_h = h + delta_h.T
        x = self.decode(perturbed_h)
        return x

    def sample_scan(self, x, sigma, n_steps, samples):
        # enable on-the-fly graph computations
        # theano.config.compute_test_value = 'raise'
        in_val = T.fmatrix("input_values")
        #in_val.tag.test_value = numpy.asarray(numpy.random.rand(1, 784), dtype=theano.config.floatX)
        s_sigma = T.fscalar("sigma_values")
        #s_sigma = numpy.asarray(numpy.random.rand(1), dtype=theano.config.floatX)
        mode = "FAST_RUN"
        values, updates = theano.scan(fn=self.sample_one_step,
            outputs_info=in_val,
            non_sequences=s_sigma,
            n_steps=n_steps,
            mode=mode)
        ae_sampler = theano.function(inputs=[in_val, s_sigma], outputs=values[-1], updates=updates)
        samples = ae_sampler(x, sigma)
        return samples

    def sample_old(self, x, sigma, n_steps):
        # enable on-the-fly graph computations
        # theano.config.compute_test_value = 'raise'
        #in_val = T.fmatrix("input_values")
        #in_val.tag.test_value = numpy.asarray(numpy.random.rand(1, 784), dtype=theano.config.floatX)
        #s_sigma = T.fscalar("sigma_values")
        #s_sigma = numpy.asarray(numpy.random.rand(1), dtype=theano.config.floatX)
        #mode = "FAST_RUN"
        samples = []
        sample = x
        samples.append(x)
        for i in xrange(n_steps):
            print "Sample %d..." % i
            sampler = self.sample_one_step(sample, sigma)
            sample = sampler.eval()
            samples.append(sample)
        return samples

    def get_sgd_updates(self, learning_rate, lr_scaler=1.0, batch_size=-1, sparsity_level=-1, sparse_reg=-1, x_in=None):
        h = self.encode(x_in)
        x_rec = self.decode(h)
        cost = self.get_rec_cost(x_rec)

        if self.L1_reg != -1 and self.L1_reg is not None:
            cost += self.L1_reg * self.L1

        if self.L2_reg != -1 and self.L2_reg is not None:
            cost += self.L2_reg * self.L2

        if sparsity_level != -1 and sparse_reg != -1:
            sparsity_penal = self.sparsity_penalty(h, sparsity_level, sparse_reg, batch_size)
            cost += sparsity_penal

        self.gparams = T.grad(cost, self.params)
        updates = OrderedDict({})
        for param, gparam in zip(self.params, self.gparams):
            updates[param] = self.momentum * param - lr_scaler * learning_rate * gparam
        return (cost, updates)

    def fit(self,
            data=None,
            learning_rate=0.1,
            batch_size=100,
            n_epochs=20,
            lr_scaler=0.998,
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
        (cost, updates) = self.get_sgd_updates(learning_rate, lr_scaler, batch_size)
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
