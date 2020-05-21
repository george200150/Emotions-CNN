'''
Created on 10 mai 2020

@author: George
'''
import numpy as np
from math import floor
import cv2
from random import random

def writeHappyToFile():
    f = open('emotions.csv','w')
    f.write('\n') #first line is ignored anyways
    g = open('happy.txt')
    
    while True:
        fileName = g.readline().strip()
        
        if len(fileName) == 0:
            break
        
        img = cv2.imread('resized/'+fileName, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        rnd = random()
        if rnd > 0.6 and rnd <= 0.8:
            writeBitmap(1,'PublicTest',f,img)
        elif rnd > 0.8:
            writeBitmap(1,'PrivateTest',f,img)
        else:
            writeBitmap(1,'Training',f,img) # 1 for happy; 0 for sad
    g.close()
    f.close()


def writeSadToFile():
    f = open('emotions.csv','a')
    g = open('sad.txt')
    
    while True:
        fileName = g.readline().strip()
        
        if len(fileName) == 0:
            break
        
        img = cv2.imread('resized/'+fileName, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        rnd = random()
        if rnd > 0.6 and rnd <= 0.8:
            writeBitmap(0,'PublicTest',f,img)
        elif rnd > 0.8:
            writeBitmap(0,'PrivateTest',f,img)
        else:
            writeBitmap(0,'Training',f,img) # 1 for happy; 0 for sad
    g.close()
    f.close()


def writeBitmap(clas,typ,f,img):
    to_write = ''
    to_write += str(clas)
    to_write += ','
    to_write += typ
    to_write += ','
    for line in img:
        for gray_pixel in line:
            to_write += str(gray_pixel)
            to_write += ' '
    to_write = to_write[:-1]
    to_write += '\n'
    f.write(to_write)


def load_data():
    f = open('emotions.csv')
    f.readline()
    training_inputs = []
    training_outputs = []
    
    validation_inputs = []
    validation_outputs = []
    
    test_inputs = []
    test_outputs = []
    
    
    while True:
        
        line = f.readline()
        parts = line.strip().split(',') # emotion, data_type, string_byte_pixels
        
        if len(parts) < 2:
            break
        
        bitmap = parts[2].split(' ')
        #bitmap = [int(x)/255 for x in bitmap]
        bitmap = [floor(int(x)/255+0.5) for x in bitmap] 
        if parts[1] == 'Training':
            training_inputs.append(bitmap)
            training_outputs.append(int(parts[0]))
            pass
        elif parts[1] == 'PublicTest':
            validation_inputs.append(bitmap)
            validation_outputs.append(int(parts[0]))
            pass
        elif parts[1] == 'PrivateTest':
            test_inputs.append(bitmap)
            test_outputs.append(int(parts[0]))
            pass
    
    
    training_inputs = [np.asarray(x, np.float32).reshape([784,1]) for x in training_inputs]
    validation_inputs = [np.asarray(x, np.float32).reshape([784,1]) for x in validation_inputs]
    test_inputs = [np.asarray(x, np.float32).reshape([784,1]) for x in test_inputs]
    
    
    
    training_inputs = np.reshape(training_inputs, (len(training_inputs),784))
    training_outputs = np.reshape(training_outputs, -1) # automatically infer dimension
    training_data = (training_inputs, training_outputs)
    
    validation_inputs = np.reshape(validation_inputs, (len(validation_inputs),784))
    validation_outputs = np.reshape(validation_outputs, -1) # automatically infer dimension
    validation_data = (validation_inputs, validation_outputs)
    
    test_inputs = np.reshape(test_inputs, (len(test_inputs),784))
    test_outputs = np.reshape(test_outputs, -1) # automatically infer dimension
    testing_data = (test_inputs, test_outputs)

    
    return training_data, validation_data, testing_data


if __name__ == '__main__':
    writeHappyToFile()
    writeSadToFile()