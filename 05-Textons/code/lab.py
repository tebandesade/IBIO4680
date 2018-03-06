import sys
import os
import numpy as np
#Load sample images from disk
from skimage import color
from skimage import io
import pickle as pkl
sys.path.append('lib/python')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.ensemble import RandomForestClassifier

def get_subFiles(p,r):
    imagenes = os.listdir(p)
    indices_mezclados = np.random.permutation(len(imagenes))
    for i in indices_mezclados[:10]:
        rta = os.path.join(p,imagenes[i])
        test_local.append(rta)
        test_cat_mapping[rta] = r
    for i in indices_mezclados[10:]:
        rta = os.path.join(p,imagenes[i])
        train_local.append(rta)
        train_cat_mapping[rta]=r
    
def get_subFolderFolder(rt,sb):
    for  r in (rt):
        categories.append(r)
        rutaruta = os.path.join(sb,r)
        get_subFiles(rutaruta,r)
    
def get_subFolders(ruta):
    for ru in ruta:
        sub_ruta = os.path.join(data_path,ru)
        sub = os.listdir(sub_ruta)
        #print(sub)
        get_subFolderFolder(sub,sub_ruta)
def create_texton_dict():
    tmp = {}
    for cat in categories:
        tmp[cat]=[]
    return tmp

def configure(list_,flag_):
    temp = []
    temp_dict = {}
    for im in list_:
        
        arra = io.imread(im)
        re_shape = arra[:30,30:60]
        temp_dict[im] = re_shape
        if flag_ ==0:
            ct  = train_cat_mapping[im]
            txtn_train_dict[ct].append(re_shape)
        else:
            ct  = test_cat_mapping[im]
            txtn_test_dict[ct].append(re_shape)
        temp.append(re_shape)
    return temp,temp_dict

def asignar_texton_train(dc):
    coter = 0
    tm_dict = {}
    for k,v in dc.items():
        tmp_rr = []
        for ele in v:
            tmap =assignTextons(fbRun(fb,ele),textons.transpose())
            tmp_rr.append(tmap)
        tm_dict[k]=tmp_rr
    return tm_dict
def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

def create_data_ml(dicti):
    data_ = []
    labels_ = []
    for k, v in dicti.items():
        label_ = k
        arra   = v
        for ar in arra:
            h = histc(ar.flatten(), np.arange(25))/len(ar)
            data_.append(h)
            labels_.append(k)
    return data_, labels_

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imsave('confusionmatrix.jpg',cm)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=90)
    #plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    diag = np.diag(cm)
    acc = np.average(diag)
    return acc

data_path = '../data/'
data_dir  = os.listdir(data_path)
test_local = []
train_local =[]
categories = []
train_cat_mapping = {}
test_cat_mapping  = {}
np.random.seed(42)

get_subFolders(data_dir)

test_dict = {}
txtn_train_dict = create_texton_dict()
txtn_test_dict  = create_texton_dict()

list_images_entrenamiento , train_dict = configure(train_local,0)
images_test, test_dict = configure(test_local,1)

from fbCreate import fbCreate
fb = fbCreate()
hstac = np.hstack(list_images_entrenamiento)

from fbRun import fbRun
filterResponses = fbRun(fb,np.hstack(list_images_entrenamiento))

from computeTextons import computeTextons
from assignTextons import assignTextons
map, textons = computeTextons(filterResponses, 25)

txtn_train_dict = asignar_texton_train(txtn_train_dict)
txtn_test_dict = asignar_texton_train(txtn_test_dict)

entrenar_x, entrenar_y = create_data_ml(txtn_train_dict)


probar_x, probar_y = create_data_ml(txtn_test_dict)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(entrenar_x,entrenar_y)

  #Predict the train data
result1 = clf.predict(entrenar_x)
mtrx = confusion_matrix(entrenar_y,result1)
average = plot_confusion_matrix(mtrx, classes=list(txtn_train_dict.keys()),normalize=True, title='Train Confusion matrix, with normalization')
plt.show()
print('Training nn average classification error ',average)

clf2 = KNeighborsClassifier()
clf2.fit(probar_x,probar_y)
result2 = clf.predict(probar_x)
mtrx_2 = confusion_matrix(probar_y,result2)
average_2 = plot_confusion_matrix(mtrx_2, classes=list(txtn_test_dict.keys()),normalize=True, title='Test Confusion matrix, with normalization')
print('Testing nn average classification error ',average_2)
plt.show()

rndomforrest = RandomForestClassifier(n_estimators=10)
rndomforrest.fit(entrenar_x,entrenar_y)
#Predict the train data
predi_train = rndomforrest.predict(entrenar_x)
rnd_mtrx = confusion_matrix(entrenar_y,predi_train)
average_3 = plot_confusion_matrix(rnd_mtrx, classes=list(txtn_train_dict.keys()),normalize=True, title='Train Confusion matrix, with normalization')

print('Training random forest average classification error ',average_3)


rndomforrest2 = RandomForestClassifier(n_estimators=10)
rndomforrest2.fit(entrenar_x,entrenar_y)
#Predict the train data
predi_train2 = rndomforrest2.predict(probar_x)
rnd_mtrx = confusion_matrix(probar_y,predi_train2)
average_4 = plot_confusion_matrix(rnd_mtrx, classes=list(txtn_test_dict.keys()),normalize=True, title='Train Confusion matrix, with normalization')

print('Testing rndm forest average classification error ',average_4)