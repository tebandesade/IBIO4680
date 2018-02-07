import cv2
import os
import sys
#The input directory has to be data/images 
path_ = sys.argv[1]

for dire in os.listdir(path_):
	ruta = path_+'/'+dire
	for pic in os.listdir(ruta):
		if '.jpg' in pic:
			ruta_pic = ruta + '/'+pic
			img = cv2.imread(ruta_pic)
			new_cropped_im = cv2.resize(img,(256,256))
			print(str(pic)+'/'+str(new_cropped_im.shape))
