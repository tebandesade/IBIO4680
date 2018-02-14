import requests
import tarfile
import sys
import os 
import scipy
import cv2
from matplotlib import pyplot as plt
from scipy import io
import numpy as np 
import time

plt.switch_backend('agg')

#Choose n number of pictures
n = sys.argv[1]
t_inicial = time.time()
#Tar file must be in same directory as script
tar_file = 'BSR_bsds500.tgz'
tar = tarfile.open(tar_file,'r')
tar.extractall()
tar.close()
#tar file has been extracted, now lets load some images!

#Lets choose some N random images 
images_escogidas = np.random.randint(0,size_images,int(n))
imagenes_finales = set(images_escogidas)
tam_random = len(imagenes_finales)

#Remember random can generate duplicates 
if tam_random < int(n):
    print('Random genero imagenes duplicadas :(')
os.mkdir('out')

diccionario = {}
for index in imagenes_finales:
    ori  = plt.imread(train_images+str(train_orfiles[index]))
    ori  = cv2.resize(ori, (256,256))
    tes  = io.loadmat(ground_truth_train+str(train_grfiles[index]))
    grou = tes['groundTruth']
    plt.imshow(ori)
    plt.imsave('out/'+str(train_orfiles[index])+str(ind)+'-seg.png',ori)
    plt.show()
    obj = []
    for ind, i in enumerate(grou[0]): 
        #print(train_grfiles[index])
        #print(ind)
        ch_s = cv2.resize( i[0][0][0], (256, 256))
        plt.imshow(ch_s,cmap='gray')
        plt.imsave('out/'+str(train_grfiles[index])+str(ind)+'-seg.png',ch_s)
        plt.show()
        ch_b = cv2.resize( i[0][0][1], (256, 256))
                            
        plt.imshow(ch_b,cmap='gray')
        plt.imsave('out/'+str(train_grfiles[index])+str(ind)+'-bou.png',ch_b)
        plt.show()
        obj.append('out/'+str(train_grfiles[index])+str(ind)+'-seg.png')
        obj.append('out/'+str(train_grfiles[index])+str(ind)+'-bou.png')
    diccionario['out/'+str(train_orfiles[index])+str(ind)+'-seg.png'] = obj
    
np.save('diccionario_ims.npz',diccionario)
fina_time = time.time() - t_inicial
print('Tiempo final es: ' + str(fina_time))
