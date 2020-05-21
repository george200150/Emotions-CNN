'''
Created on 19 mai 2020

@author: George
'''
# https://emojipedia.org/people/
from network import csv_load_data_shared
from network import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network import ReLU
from myLoader import load_data
import numpy as np

def testNet(net):
    training_data, validation_data, testing_data = load_data()
    nparrayBitmap = testing_data[0][0]
    true = testing_data[1][0]
    computed = net.feedforward(nparrayBitmap)
    print("The real result is {} and the computed result is {}".format(true,computed))
    
    nparrayBitmap = testing_data[0][3]
    true = testing_data[1][3]
    computed = net.feedforward(nparrayBitmap)
    print("The real result is {} and the computed result is {}".format(true,computed))
    
    nparrayBitmap = training_data[0][20]
    true = np.argmax(training_data[1][20])
    computed = net.feedforward(nparrayBitmap)
    
    print("The real result is {} and the computed result is {}".format(true,computed))


if __name__ == '__main__':
    training_data, validation_data, testing_data = csv_load_data_shared()
    mini_batch_size = 2
    
    
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=300, activation_fn=ReLU),
        FullyConnectedLayer(n_in=300, n_out=50, activation_fn=ReLU),
        SoftmaxLayer(n_in=50, n_out=2)], mini_batch_size)
    
    net.SGD(training_data, 15, mini_batch_size, 0.1, validation_data=validation_data, test_data=testing_data, lmbda=3.0)
    
    net.saveWholeNet()
    
    
    testNet(net)