'''
Created on 20 apr. 2020

@author: George
'''
import numpy as np
from math import floor, sqrt
import cv2
from skimage import img_as_ubyte, img_as_float32
from cv2.cv2 import IMREAD_GRAYSCALE

def convertListToNDArray(lista):
    lista = np.asarray(lista, dtype='uint8')
    size = floor(sqrt(len(lista)))
    lista = lista.reshape([size,size])
    #lista = lista * 255
    #lista = ~lista
    return lista

def reader(fileName, only=None, without=None):
    '''
    @param fileName: string name of the source csv file
    @param only: function (None by default -> all dataset is loaded) to consider only specific fonts or characters
    e.g. only=lambda x:x.contains("italic")
    @param without: function (None by default -> all dataset is loaded) to ignore fonts or characters
    e.g. without=lambda x:x.contains("serif")
    @return: two lists, the first containing the characters and the second containing the lists of pixels
    '''
    f = open(fileName, 'r')
    
    characters = []
    bitmaps = []
    
    while True:
        line = f.readline()
        parts = line.split(',') # parts[0] == font ; parts[1] == character ; parts[2:] = pixels ---> 785 commas each line - 172732 lines total
        
        if len(parts) < 784: # stop condition is that the line is empty
            f.close();
            print("RETURNED len({}) is {} and len({}) is {}".format("characters", len(characters),"bitmaps",len(bitmaps)))
            total = len(bitmaps[-1])
            size = floor(sqrt(total))
            print("A bitmap has {} pixels, wich is {} by {} pixels".format(total,size,size))
            return characters, bitmaps
        
        if without != None:
            if without(parts):
                append(characters, bitmaps, parts)
        
        elif only != None:
            if only(parts):
                append(characters, bitmaps, parts)
        
        else:
            append(characters, bitmaps, parts)


def append(characters, bitmaps, parts):
    '''
    the parameters are pointers to the lists of characters, pixel maps and the line split in parts of the file
    (avoid duplicate code)
    '''
    characters.append(parts[1])
    ints = [int(x) for x in parts[2:]]
    bitmaps.append(ints)

def are_italics(parts):
    font = parts[0][:-4] # remove the .ttf of the font name
    components = font.split("-")
    for tag in components:
        if 'italic' in tag:
            return False
    return True



def reverseMapCodeToASCII(code): # https://www.kaggle.com/crawford/emnist#emnist-balanced-mapping.txt
    mapper = {48:0, 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9, 65:10,
    66:11, 67:12, 68:13, 69:14, 70:15, 71:16, 72:17, 73:18, 74:19, 75:20, 76:21,
    77:22, 78:23, 79:24, 80:25, 81:26, 82:27, 83:28, 84:29, 85:30, 86:31, 87:32, 88:33, 89:34,
    90:35, 97:36, 98:37, 99:38, 100:39, 101:40, 102:41, 103:42, 104:43, 105:44, 106:45, 107:46,
    108:47, 109:48, 110:49, 111:50, 112:51, 113:52, 114:53, 115:54, 116:55, 117:56, 118:57, 119:58,
    120:59, 121:60, 122:61}
    return mapper[code]

def loaderForTensor(trainPercent, _):
    #chars, bitmaps = reader('small.txt', are_italics, None)
    chars, bitmaps = reader('../emotions.csv', are_italics, None) # consider only italics
    init_size = int(floor(sqrt(len(bitmaps[0]))))
    
    bitmapsEnhanced = []
    for bitmap in bitmaps:
        bitmap = np.asarray(bitmap).reshape([init_size,init_size])
        bitmap = ~img_as_ubyte(bitmap)
        cv2.imwrite('temp.png',bitmap) 
        bitmap = cv2.imread('temp.png', IMREAD_GRAYSCALE)
        after_size = len(bitmap)
        print(after_size)
        #bitmap = img_as_float32(bitmap)
        #bitmap = np.asarray([[pixel,pixel,pixel] for pixel in bitmap]).reshape([after_size,after_size,3])
        bitmapsEnhanced.append(bitmap)
    
    
    size = len(chars)
    splitter = size * trainPercent // 100
    train_chars, train_bitmaps = chars[:splitter], bitmapsEnhanced[:splitter]
    test_chars, test_bitmaps = chars[splitter:], bitmapsEnhanced[splitter:]
    
    train_chars = np.asarray([reverseMapCodeToASCII(ord(c)) for c in train_chars]).reshape([-1,1])
    train_bitmaps = [np.asarray([[pixel,pixel,pixel] for pixel in bitmap]).reshape([after_size,after_size,3]) for bitmap in train_bitmaps] # mimic RGB channels
    train_bitmaps = np.asarray(train_bitmaps)
    
    test_chars = np.asarray([reverseMapCodeToASCII(ord(c)) for c in test_chars]).reshape([-1,1])
    test_bitmaps = [np.asarray([[pixel,pixel,pixel] for pixel in bitmap]).reshape([after_size,after_size,3]) for bitmap in test_bitmaps] # mimic RGB channels
    test_bitmaps = np.asarray(test_bitmaps)
    return train_bitmaps, train_chars, test_bitmaps, test_chars