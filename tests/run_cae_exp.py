from ca import ContractiveAutoencoder
#from dataset import Dataset

import theano
#import theano.tensor as T
import numpy
import cPickle as  pkl

cae_fhdlr = open("cae_data.pkl", 'rb')
mnist_fhdlr = open("/data/lisa/data/mnist/mnist_all.pickle", 'rb')

cae = pkl.load(cae_fhdlr)
mnist = pkl.load(mnist_fhdlr)

data = numpy.asarray(mnist[0][20].reshape(1, 784), dtype=theano.config.floatX)
#data = numpy.asarray(numpy.random.rand(1, 784), dtype=theano.config.floatX)
#samples = []

#samples.append(data)
#last_sample = data
#for i in xrange(99):
#last_sample = cae.sample(last_sample, 0.009, 1000)
#    samples.append(last_sample)
samples = cae.sample(data, 0.08, 10000)
#samples.insert(0, data)
numpy.save(open("data_samples.npy", 'wb'), samples)
#samples = cae.sample(numpy.asarray(data, dtype=theano.config.floatX), sigma=0.3, n_steps=100)
