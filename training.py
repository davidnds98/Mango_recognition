import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os    
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def traindata(size,RGB):        #(圖片尺寸與RGB通道);(120,3)
    
    #讀取圖片
    TrainPath = './Second/test_train/'
    TrainallFileList = os.listdir(TrainPath)
    #讀取csv資料
    train = np.genfromtxt('./Second/train_div.csv', delimiter=',', dtype=None)[1:]
    trainid = np.array([label for label,image_id in train]).astype(np.str)
    trainlabel = np.array([image_id for label,image_id in train]).astype(np.str)
    
    trainX=[]
    trainY=[]
    for i in range(len(TrainallFileList)):
        trainX.append(Pretreatment(TrainPath+trainid[i],size))
        trainY.append(replacelabel(trainlabel[i]))
    X_Train=np.array(trainX)
    y_Train=np.array(trainY)
    X_Train40 = X_Train.reshape(X_Train.shape[0], size,size,RGB).astype('float32')
    
    
    #標準化特徵
    X_Train40_norm = X_Train40 / 255
        
    #Onehot-encoding
    y_TrainOneHot = np_utils.to_categorical(y_Train)
    return X_Train40_norm,y_TrainOneHot

def Pretreatment(path,size) :    
    img = plt.imread(path)
    
    #resize
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)    
    arr=np.array(img)
    return arr

def replacelabel(label):
    if(label=='不良-著色不佳'):
        return 0
    elif (label=='不良-乳汁吸附'):
        return 1
    elif (label=='不良-炭疽病'):
        return 2
    elif (label=='不良-黑斑病'):
        return 3
    elif(label=='不良-機械傷害'):
        return 4
    
def makemodel(X_Train40_norm,y_TrainOneHot,size,RGB):
    
    model = Sequential()
    
    model.add(Conv2D(filters=8,kernel_size=(3, 3),padding='same',input_shape=(size, size, RGB),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16,kernel_size=(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64,kernel_size=(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128,kernel_size=(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128,kernel_size=(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Flatten())
    
    # Fully connected
    model.add(Dense(5, activation='softmax'))
    model.summary()
    print("")
    
    #檔名設定
    epochs=20
    batch_size=128
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    train_history = model.fit(x=X_Train40_norm, y=y_TrainOneHot, epochs=epochs, batch_size=batch_size, verbose=1,
    validation_split=0.2, shuffle=True)
    model.save('mango120_model.h5')

    # show_train_history(train_history, 'acc', 'val_acc')
    # show_train_history(train_history, 'loss', 'val_loss')    

def show_train_history(train_history, train, validation):
    # plot train set accuarcy / loss function value ( determined by what parameter 'train' you pass )
    # The type of train_history.history is dictionary (a special data type in Python)
    
    plt.plot(train_history.history[train])
    # plot validation set accuarcy / loss function value
    plt.plot(train_history.history[validation])
    # set the title of figure you will draw
    plt.title('Train History')
    # set the title of y-axis
    plt.ylabel(train)
    # set the title of x-axis
    plt.xlabel('Epoch')
    # Places a legend on the place you set by loc
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    
    plt.subplot(121)
    plt.plot(loss)
    plt.subplot(122)
    plt.plot(val_loss)
    plt.show()
    
#讀取csv資料
TrainPath = './Second/test_train/'
TrainallFileList = os.listdir(TrainPath)

train = np.genfromtxt('D:/NTNU/Pattern recognition/final project/Second/train_div.csv', delimiter=',', dtype=None)[1:]
trainid = np.array([label for label,image_id in train]).astype(np.str)
trainlabel = np.array([image_id for label,image_id in train]).astype(np.str)

#設定圖片尺寸與RGB通道
size=120
RGB=3

#訓練資料前處理
#X_Train40_norm,y_TrainOneHot=traindata(size,RGB)
A=np.load('D:/NTNU/Pattern recognition/final project/Second/mango_size90.npz')

#儲存npz圖片
#np.savez('./Second/mango_size120.npz',X_Train40_norm=X_Train40_norm,y_TrainOneHot=y_TrainOneHot)

#讀取npz圖片
X_Train40_norm=A['X_Train40_norm']
y_TrainOneHot=A['y_TrainOneHot']

#模型訓練
makemodel(X_Train40_norm,y_TrainOneHot,size,RGB)





