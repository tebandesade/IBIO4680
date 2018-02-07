import cv2
import os
import sys
#The input directory has to be data/images 
path_ = sys.argv[1]

port = 0
land = 0
for dire in os.listdir(path_):
	ruta = path_+'/'+dire
	for pic in os.listdir(ruta):
		if '.jpg' in pic:
			ruta_pic = ruta + '/'+pic
			img = cv2.imread(ruta_pic)
			h, w, rgb = img.shape
			if h>w:
				#print(str(pic)+' is portrait')
				port +=1
			else:
				land +=1
				#print(str(pic)+' is landscape')

print('Portrait: ',port)
print('Landscape: ', land)