import os
import numpy as np 
from PIL import Image
import cv2
import hashlib

def check_if_duplicate(hash_,im):
    if hash_ in dicti.keys():
        return True
    else:
        dicti[hash_] = im
        return False
def find_duplicates(group):
    name = group[0]
    for pic in group[1]:
        path = name+'/'+pic
        image = cv2.imread(path)
        
        if len(image.shape)>2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hash_ = hashlib.md5(image)
            ret = check_if_duplicate(hash_,image)
            print(ret)
        else:
            hash_ = hashlib.md5(image)
            ret = check_if_duplicate(hash_,image)
            print(ret)

dicti = {}
aerials = os.listdir('aerials')
misc    = os.listdir('misc')
sequences = os.listdir('sequences')
textures  = os.listdir('textures')
data_set  = {}
data_set['aerials'] = aerials
data_set['misc']    = misc
data_set['sequences'] = sequences
data_set['textures']  = textures


for item in data_set.items():    
    find_duplicates(item)