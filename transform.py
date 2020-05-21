'''
Created on 19 mai 1010

@author: George
'''
import cv2
import numpy as np
from PIL import Image

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)




def contrasting(photoName, outName):
    '''
    @param photoName: input photo file
    @param outName: output photo file 
    '''
    img = Image.open(photoName)
    img.load()
    
    result = change_contrast(img, 210) # 258 is maximum
    result.save(outName)
    print('done')




def reshape_dataset_only_once_call_ever():
    """
    load the pokemon images from "images" directory based on the names written on each line of "pokemon.csv"
    and reshapes all the images into 28x28 images and saves normal and sepia versions of each in "resized" directory.
    """
    
    f = open('happy.txt', 'r')
    line = f.readline() # ignore first line
    while True:
        line = f.readline()
        parts = line.strip().split(",")
        if len(parts) <= 0:
            break
        #contrasting('EMOJI/HAPPY/'+parts[0], "temp.png")
        #img = cv2.imread("temp.png")
        img = cv2.imread('EMOJI/HAPPY/'+parts[0])
        
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        
        dim = (28, 28)
        #dim = (100, 100)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        
        lower_white = np.array([5, 5, 5], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(resized, lower_white, upper_white) # could also use threshold
        coloured = resized.copy()
        coloured[mask == 255] = (255, 255, 255)
        
        cv2.imwrite('resized/'+parts[0], coloured)
    f.close()

def reshape_dataset_only_once_call_ever2():
    f = open('sad.txt', 'r')
    line = f.readline() # ignore first line
    while True:
        line = f.readline()
        parts = line.strip().split(",")
        if len(parts) <= 0:
            break
        #contrasting('EMOJI/SAD/'+parts[0], "temp.png")
        #img = cv2.imread("temp.png")
        img = cv2.imread('EMOJI/SAD/'+parts[0])
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        dim = (28, 28)
        #dim = (100, 100)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        
        
        """lower_white = np.array([0, 0, 0], dtype=np.uint8)
        upper_white = np.array([20, 20, 20], dtype=np.uint8)"""
        lower_white = np.array([5, 5, 5], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(resized, lower_white, upper_white) # could also use threshold
        coloured = resized.copy()
        coloured[mask == 255] = (255, 255, 255)
        
        cv2.imwrite('resized/'+parts[0], coloured)
    f.close()
    pass






if __name__ == '__main__':
    try:
        reshape_dataset_only_once_call_ever()
    except Exception:
        pass
    try:
        reshape_dataset_only_once_call_ever2()
    except Exception:
        pass
    
    