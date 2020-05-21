'''
Created on 10 mar. 2020

@author: George
'''
from theano.scalar.basic import float32
from pickle import HIGHEST_PROTOCOL
from copy import deepcopy
#TODO: RUN -> RUN CONFIGURATIONS -> INTERPRETER -> CHOOSE PYTHONx86 -> RUN (...compile error for python x64...)





#from CSV_Loader import load_data
from myLoader import load_data

import pickle
import gzip



import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample




def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid


#### Constants
GPU = True
theano.config.cxx = 'C:/MinGW/bin/g++.exe'              # x86
#theano.config.cxx = 'C:/Users/George/bin/g++.exe'      # x64
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify network.py to set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    theano.config.device = 'cpu'
    print("Running with a CPU.  If this is not desired, then the modify network.py to set the GPU flag to True.")



def csv_load_data_shared():
    training_data, validation_data, test_data = load_data()
    
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)
    
    
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        #theano.config.floatX = 'float32'
        shared_x = theano.shared(
            np.asarray(data[0], dtype='float32'), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype='float32'), borrow=True)
        return shared_x, T.cast(shared_y, 'int32')
    
    return [shared(training_data), shared(validation_data), shared(test_data)]


def make_shared(nparrayBitmap):
    data = ([nparrayBitmap],[-1])
    
    def shared(data):
            """Place the data into shared variables.  This allows Theano to copy
            the data to the GPU, if one is available.
            """
            #theano.config.floatX = 'float32'
            shared_x = theano.shared(
                np.asarray(data[0], dtype=float32), borrow=True)
            shared_y = theano.shared(
                np.asarray(data[1], dtype=float32), borrow=True)
            return shared_x, T.cast(shared_y, "int32")
    
    return shared(data)

def shared_predict(nparrayBitmap):
    test_data = ([nparrayBitmap,nparrayBitmap],[-1,-1])
    
    def shared(data):
            """Place the data into shared variables.  This allows Theano to copy
            the data to the GPU, if one is available.
            """
            #theano.config.floatX = 'float32'
            shared_x = theano.shared(
                np.asarray(data[0], dtype=float32), borrow=True)
            shared_y = theano.shared(
                np.asarray(data[1], dtype=float32), borrow=True)
            return shared_x, T.cast(shared_y, "int32")
    return shared(test_data)

def load_data_shared():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        #theano.config.floatX = 'float32'
        shared_x = theano.shared(
            np.asarray(data[0], dtype=float32), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=float32), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    # TypeError: Cannot convert Type TensorType(int64, vector) (of Variable Subtensor{int64:int64:}.0) into Type TensorType(int32, vector). You can try to manually convert Subtensor{int64:int64:}.0 into a TensorType(int32, vector).
    
    return [shared(training_data), shared(validation_data), shared(test_data)]



def loadWholeNet():
    f = gzip.open('ENET.pkl.gz', 'rb')
    net = pickle.load(f, encoding="latin1")
    f.close()
    return net


class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
    
    
    def feedforward(self, nparrayBitmap):
        '''
        SO HOW DOES THIS ALL FREAKING THING WORK, YOU ASK?
        - it must use a Theano function. (such as accuracy, cost...)
        - we give the function the x
        - x is the bitmap
        - unfortunatelly, we must create a list of "minibatch_size" length of x-es
        - we apply the feedforward function of the net
        (HAVING AS INPUT (givens = self.x:test_x[...]) OUR BITMAP, WE CAN USE THE OUTPUT OF THE LAST LAYER 
        OF THE NETWORK IN ORDER TO GET THE CLASSIFICATION RESULT. (which is also a vector of the same size)
        Therefore, we get only the first element of the result vector (all are the same, of course) and
        we return the PREDICTED CLASS.)
        
        TODO: optimise algo.
        IN ORDER TO DO SO, WE WILL CHANGE THE WHOLE INPUT/OUTPUT DIMENSIONS OF
        ALL THE LAYERS TO A SINGLE VALUE. (and create a separate function - a pure net.predict())
        
        reset all set_inpt.
        call single valued theano function.
        done. hopefully.
        '''
        test_data1 = shared_predict(nparrayBitmap)
        
        test_x, _ = test_data1
        interval = self.mini_batch_size
        
        i = T.lscalar() # mini-batch index
        y_predict_theano = theano.function([i],self.layers[-1].y_out,
            givens={self.x:test_x[i*interval : (i+1)*interval]}) # CNN must be able to reshape - length of mini_batch !!!
        
        #the whole network is designed to work only with batches.
        #therefore, we will use arrays of size 10 of the same element.
        #(not so unefficient... it's pretty fast - or we could change the net..)
        #
        
        ypredict = y_predict_theano(0) # PREDICTS THE OUTPUT (in a ndarray)
        
        corect = ypredict[0]
        return corect
        
    def saveWholeNet(self):
        f = gzip.open('ENET.pkl.gz', 'w')
        pickle._dump(self, f, protocol=HIGHEST_PROTOCOL)
        f.close()
    
    

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)//mini_batch_size
        num_validation_batches = size(validation_data)//mini_batch_size
        num_test_batches = size(test_data)//mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function([i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        
        
        self.test_mb_predictions = theano.function( #TODO @THIS IS GOING BIG!
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        # Do the actual training
        #best_iteration = -1
        #test_accuracy = -1
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=float32),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=float32),
            borrow=True)
        self.params = [self.w, self.b]
    

        
    

    #from theano.tensor.signal.pool import pool_2d for higher versions of Theano

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape,image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=float32),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=float32),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        


    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=float32),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=float32),
            name='b', borrow=True)
        self.params = [self.w, self.b]
    


    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, 'float32')
