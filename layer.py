from __future__ import division
import numpy
import theano
from theano import tensor as T

class Layer(object):
    """
     A general base layer class for neural networks.
    """
    def __init__(self,
        input,
        n_in,
        n_out,
        activation=T.nnet.sigmoid,
        sparse_initialize=False,
        num_pieces=1,
        non_zero_units=25,
        rng=None):

        self.num_pieces = num_pieces
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.sparse_initialize = sparse_initialize
        self.non_zero_units = non_zero_units
        self.W = None
        self.b = None
        self.sparse_initialize = sparse_initialize
        self.activation = activation

    def reset_layer(self):
        if self.W is None:
            if self.sparse_initialize:
                W_values = self.sparse_initialize_weights()
            else:
                W_values = numpy.asarray(self.rng.uniform(
                    low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)),
                    dtype=theano.config.floatX)

                if self.activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

        if self.b is None:
            b_values = numpy.zeros((self.n_out/self.num_pieces), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        # parameters of the model
        self.params = [self.W, self.b]

    def sparse_initialize_weights(self):
        #Implement the sparse initialization technique as decribed in 2010 Martens.
        W = []
        mu, sigma = 0, 1/self.non_zero_units

        for i in xrange(self.n_in):
            row = numpy.zeros(self.n_out)
            non_zeros = self.rng.normal(mu, sigma, self.non_zero_units)
            #non_zeros /= non_zeros.sum()
            non_zero_idxs = self.rng.permutation(self.n_out)[0:self.non_zero_units]
            for j in xrange(self.non_zero_units):
                row[non_zero_idxs[j]] = non_zeros[j]
            W.append(row)
        W = numpy.asarray(W, dtype=theano.config.floatX)
        return W

class AEHiddenLayer(Layer):

    def __init__(self,
            input,
            n_in,
            n_out,
            n_in_dec=None,
            n_out_dec=None,
            W=None,
            b=None,
            num_pieces=1,
            bhid=None,
            activation=T.nnet.sigmoid,
            sparse_initialize=False,
            tied_weights=True,
            rng=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int

        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = numpy.random.RandomState()

        super(AEHiddenLayer, self).__init__(input,
                n_in,
                n_out,
                num_pieces=num_pieces,
                activation=activation,
                sparse_initialize=sparse_initialize,
                rng=rng)

        self.reset_layer()

        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

        if bhid is not None:
            self.b_prime = bhid
        else:
            if n_in_dec is not None:
                b_values = numpy.zeros((n_out_dec), dtype=theano.config.floatX)
            else:
                b_values = numpy.zeros((self.n_out/num_pieces), dtype=theano.config.floatX)

            self.b_prime = theano.shared(value=b_values, name='b_prime')

        if tied_weights:
            self.W_prime = self.W.T
        else:
            if n_in_dec is not None and n_out_dec is not None:
                W_values = numpy.asarray(self.rng.normal(
                    loc=0.,
                    scale=0.005,
                    size=(n_out_dec, n_in_dec)),
                    dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(self.rng.uniform(
                    low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_out, self.n_in)),
                    dtype=theano.config.floatX)

                if self.activation == T.nnet.sigmoid:
                    W_values *= 4

            self.W_prime = theano.shared(value=W_values, name='W', borrow=True)
            self.params += [self.W_prime]

        self.params += [self.b_prime]
        self.setup_outputs(input)

    def setup_outputs(self, input):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if self.activation is None
                else self.activation(lin_output))

    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output

class HiddenLayer(Layer):

    def __init__(self, input, n_in, n_out, W=None, b=None, activation=T.tanh, rng=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int

        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = numpy.random.RandomState()

        super(HiddenLayer, self).__init__(input, n_in, n_out, activation=activation, rng=rng)
        self.reset_layer()

        if W is not None:
            self.W = W

        if b is not None:
            self.b = b

        self.setup_outputs(input)

    def setup_outputs(self, input):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if self.activation is None
                else self.activation(lin_output))

    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output

class LogisticRegressionLayer(Layer):
    """
    Multi-class Logistic Regression Class.
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, input, n_in, n_out, is_binary=False, threshold=0.4, rng=None):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture
        (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which
        the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        self.activation = T.nnet.sigmoid
        super(LogisticRegressionLayer, self).__init__(input,
        n_in, n_out, self.activation, rng)

        self.reset_layer()

        self.is_binary = is_binary
        if n_out == 1:
            self.is_binary = True
        # The number of classes seen
        self.n_classes_seen = numpy.zeros(n_out)
        # The number of wrong classification made for class i
        self.n_wrong_clasif_made = numpy.zeros(n_out)

        self.reset_conf_mat()
        #
        # compute vector of class-membership probabilities in symbolic form
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = self.get_class_memberships(self.input)

        if not self.is_binary:
            # compute prediction as class whose probability is maximal in
            # symbolic form
            self.y_decision = T.argmax(self.p_y_given_x, axis=1)
        else:
            #If the probability is greater than 0.5 assign to the class 1
            # otherwise it is 0. Which can also be interpreted as check if
            # p(y=1|x)>threshold.
            self.y_decision = T.gt(T.flatten(self.p_y_given_x), threshold)

        # parameters of the model
        self.params = [self.W, self.b]

    def reset_conf_mat(self):
        """
        Reset the confusion matrix.
        """
        self.conf_mat = numpy.zeros(shape=(self.n_out, self.n_out), dtype=numpy.dtype(int))

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                    \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        if self.is_binary:
            -T.mean(T.log(self.p_y_given_x))
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def crossentropy_categorical(self, y):
        """
        Find the categorical crossentropy.
        """
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def crossentropy(self, y):
        """
        use the theano nnet cross entropy function. Return the mean.
        Note: self.p_y_given_x is (batch_size, 1) but y is is (batch_size,)
        in order to establish the compliance, we should flatten the p_y_given_x.
        """
        return T.mean(T.nnet.binary_crossentropy(T.flatten(self.p_y_given_x), y))

    def get_class_memberships(self, x):
        lin_activation = T.dot(x, self.W) + self.b
        if self.is_binary:
            """If it is binary return the sigmoid."""
            return T.nnet.sigmoid(lin_activation)
        """
            Else return the softmax class memberships.
        """
        return T.nnet.softmax(lin_activation)

    def update_conf_mat(self, y, p_y_given_x):
        """
        Update the confusion matrix with the given true labels and estimated
        labels.
        """
        if self.n_out == 1:
            y_decision = (p_y_given_x > 0.5)
            y_decision = y_decision.astype(int)
        else:
            y_decision = numpy.argmax(p_y_given_x, axis=1)
        for i in xrange(y.shape[0]):
            self.conf_mat[y[i]][y_decision[i]] +=1

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_decision, y))
        else:
            raise NotImplementedError()

    def raw_prediction_errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_decision, y)
        else:
            raise NotImplementedError()

    def error_per_classes(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_decision_res = T.neq(self.y_decision, y)
            for (i, y_decision_r) in enumerate(y_decision_res):
                self.n_classes_seen[y[i]] += 1
                if y_decision_r:
                    self.n_wrong_clasif_made[y[i]] += 1
            pred_per_class = self.n_wrong_clasif_made / self.n_classes_seen
            return T.mean(y_decision_res), pred_per_class
        else:
            raise NotImplementedError()
