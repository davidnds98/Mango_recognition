from tensorflow.keras.preprocessing import image
import glob
import tensorflow
import os
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import precision_score, recall_score,f1_score
import time
import cv2

print('Starting....')
tStart=time.time()
def excel():
    file_path='D:/NTNU/Pattern recognition/final project/Third/Test/'           #"D:\\NTNU\\Pattern recognition\\final project\\Second\\Train\\"
     #bond = pd.read_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_mangoXYWH.csv' ,header=None)       #"D:\\NTNU\\Pattern recognition\\final project\\Second\\train.csv"
    read_csv = pd.read_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_mangoXYWH.csv',encoding="big5",header=None,index_col=0)    
    #print(type(bond))
    #print(bond.iloc[0,1])
    for file_name in os.listdir(file_path): #在file_path下的檔案
        sub= os.path.splitext(file_name)    #將檔案名稱分割
        sub_name=sub[0]                     #取檔案名稱(沒有副檔名)
        print(file_name)
        a=1
        try:
            img=cv2.imread(file_path+file_name)#讀取圖片(路徑+檔名)
            """for i in read_csv.loc[file_name]:
                if(pd.isnull(read_csv.loc[file_name][a])==False):"""
            """print(read_csv.loc[file_name][a+4],'',read_csv.loc[file_name][a],'',read_csv.loc[file_name][a+1],''
                ,read_csv.loc[file_name][a+2],'',read_csv.loc[file_name][a+3])#印出read_csv的座標資料"""
        
            # 裁切區域的 x 與 y 座標（左上角）
            x=(int)(read_csv.loc[file_name][a])
            y=(int)(read_csv.loc[file_name][a+1])
            # 裁切區域的長度與寬度
            w=(int)(read_csv.loc[file_name][a+2])
            h=(int)(read_csv.loc[file_name][a+3])
            
            # 裁切圖片
            crop_img = img[y:(y+h),x:(x+w)]
            #寫入圖檔
            #cv2.imwrite('D:\\NTNU\\Pattern recognition\\final project\\Third\\testing\\'+sub_name+'.jpg', crop_img)
            newimg_path='D:\\NTNU\\Pattern recognition\\final project\\Third\\testing\\'
            
            predict(crop_img,file_name,newimg_path)
            #write_csv.loc[file_name][4]=0   #指定 D4為 0
            
            #write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet.csv', index=True, header=False) #存到csv
                #predict(newimg,file_name,file_path,bond)
        except Exception:
            pass
def predict(newimg,file_name,file_path):
    #print("check1")
    #labeltype=['不良-乳汁吸附','不良-機械傷害','不良-炭疽病','不良-著色不佳','不良-黑斑病']
    labeltype=['不良-著色不佳','不良-乳汁吸附','不良-炭疽病','不良-黑斑病','不良-機械傷害']
    #try:
    data=[]
    #img=newimg
    for Filename in os.listdir(file_path):
        data.append(Filename)
    #for i in bond.loc[file_name]:
        #trainlabel = np.array([image_id for label,image_id in train]).astype(np.str)
    pre=[]
    #print("check2")
    model = tensorflow.keras.models.load_model('D:\\NTNU\\Pattern recognition\\final project\\Second\\sigmoid mango120_model.h5')
    #print("check3")
    
    #for i in range(len(f_names)):
    images = image.load_img((file_path+file_name), target_size=(120,120,3))
    x = image.img_to_array(images)
    #print("check4")
    # Standardize feature data
    x = x / 255
    x = np.expand_dims(x, 0)
    #print('loading no.%s image' % i)
    #print("check5")
    #進行模型預測
    y = model(x)
    print(y)
    index=np.argsort(y[0,:])     #排序
    print(labeltype[index[4]],y[0,index[4]])
    pre.append(labeltype[index[4]])      #選最後的值(最大值)
    for i in range(0,5):
        if(labeltype[index[i]]>0.4 and i!=5):
            label1=labeltype[index[i]]
            label2=labeltype[index[i+1]]
            
    
    print("labeltype[index[4]]",labeltype[index[4]])        #預測的label,type=str
    write_csv = pd.read_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet2.csv',encoding="big5",header=None,index_col=0)#寫進官方的答案卷
    #print('index',index[0],'---',index[1])
    #bond.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet.csv', index=True, header=False) #存到csv
    #print("check")
    #print("write_csv.loc[file_name][1]",write_csv.loc[file_name][1],type(write_csv.loc[file_name][1]))
    #print("write_csv.loc[file_name][2]",write_csv.loc[file_name][2],type(write_csv.loc[file_name][2]))
    #print("write_csv.loc[file_name][3]",write_csv.loc[file_name][3],type(write_csv.loc[file_name][3]))
    #print("write_csv.loc[file_name][4]",write_csv.loc[file_name][4],type(write_csv.loc[file_name][4]))
    #print("write_csv.loc[file_name][5]",write_csv.loc[file_name][5],type(write_csv.loc[file_name][5]))
    if(labeltype[index[4]]=="不良-乳汁吸附"):
        print("不良-乳汁吸附")
        if(write_csv.loc[file_name][1]=="0"):
            write_csv.loc[file_name][1]=1
            write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet2.csv', index=True, header=False) #存到csv
            print("wrote")
    elif(labeltype[index[4]]=="不良-機械傷害"):
        print("不良-機械傷害")
        if(write_csv.loc[file_name][2]=="0"):
            write_csv.loc[file_name][2]=1
            write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet2.csv', index=True, header=False) #存到csv
            print("wrote")
    elif(labeltype[index[4]]=="不良-炭疽病"):
        print("不良-炭疽病")
        if(write_csv.loc[file_name][3]=="0"):
            write_csv.loc[file_name][3]=1
            write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet2.csv', index=True, header=False) #存到csv
            print("wrote")
    elif(labeltype[index[4]]=="不良-著色不佳"):
        print("不良-著色不佳")
        if(write_csv.loc[file_name][4]=="0"):
            write_csv.loc[file_name][4]=1
            write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet2.csv', index=True, header=False) #存到csv
            print("wrote")
    elif(labeltype[index[4]]=="不良-黑斑病"):
        print("不良-黑斑病")
        if(write_csv.loc[file_name][5]=="0"):
            write_csv.loc[file_name][5]=1   #指定 D5為 1
            write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_UploadSheet2.csv', index=True, header=False) #存到csv
            print("wrote")

        #print('Correct:',count)
        #print('Accuracy:',count/len(f_names))
    
        #pp=pd.crosstab(np.array(trainlabel), np.array(pre), rownames=['label'], colnames=['predict'])
    
        #print(pp)
        #print('Precision:',precision_score(trainlabel, pre, average='weighted'))
        #print('Recall:',recall_score(trainlabel, pre, average='weighted'))
        #print('f1_score:',f1_score(trainlabel, pre, average='weighted'))
    #except Exception:
        #pass
def excelandpredict():
    file_path='D:/NTNU/Pattern recognition/final project/Third/Test/'           #"D:\\NTNU\\Pattern recognition\\final project\\Second\\Train\\"
    read_csv = pd.read_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\Test_mangoXYWH.csv',encoding="big5",header=None,index_col=0)    
    labeltype=['不良-著色不佳','不良-乳汁吸附','不良-炭疽病','不良-黑斑病','不良-機械傷害']
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    for file_name in os.listdir(file_path): #在file_path下的檔案
        sub= os.path.splitext(file_name)    #將檔案名稱分割
        sub_name=sub[0]                     #取檔案名稱(沒有副檔名)
        print(file_name)
        a=1
        try:
            img=cv2.imread(file_path+file_name)#讀取圖片(路徑+檔名)
            img=cv2.filter2D(img, -1, kernel=kernel)
            # 裁切區域的 x 與 y 座標（左上角）
            x=(int)(read_csv.loc[file_name][a])
            y=(int)(read_csv.loc[file_name][a+1])
            # 裁切區域的長度與寬度
            w=(int)(read_csv.loc[file_name][a+2])
            h=(int)(read_csv.loc[file_name][a+3])
            # 裁切圖片
            crop_img = img[y:(y+h),x:(x+w)]
            #寫入圖檔
            #cv2.imwrite('D:\\NTNU\\Pattern recognition\\final project\\Third\\testing\\'+sub_name+'.jpg', crop_img)
            newimg_path='D:\\NTNU\\Pattern recognition\\final project\\Third\\blur_testing\\'
            data=[]
            for Filename in os.listdir(file_path):
                data.append(Filename)
            #for i in bond.loc[file_name]:
                #trainlabel = np.array([image_id for label,image_id in train]).astype(np.str)
            pre=[]
            #print("check2")
            model = tensorflow.keras.models.load_model('D:\\NTNU\\Pattern recognition\\final project\\Second\\sigmoid mango120_model.h5')
            #print("check3")
    
            #for i in range(len(f_names)):
            images = image.load_img((file_path+file_name), target_size=(120,120,3))
            x = image.img_to_array(images)
            #print("check4")
            # Standardize feature data
            x = x / 255
            x = np.expand_dims(x, 0)
            #print('loading no.%s image' % i)
            #print("check5")
            #進行模型預測
            y = model(x)
            print(y)
            index=np.argsort(y[0,:])     #排序
            print(labeltype[index[4]],y[0,index[4]])
            pre.append(labeltype[index[4]])      #選最後的值(最大值)
                    
            #print("labeltype[index[4]]",labeltype[index[4]])        #預測的label,type=str
            #print(label1,label2)
            write_csv = pd.read_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\sigmoid Test_UploadSheet_blur.csv',encoding="big5",header=None,index_col=0)#寫進官方的答案卷
            if(labeltype[index[4]]=="不良-乳汁吸附"):
                print("不良-乳汁吸附")
                if(write_csv.loc[file_name][1]=="0"):
                    write_csv.loc[file_name][1]=1
                    write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\sigmoid Test_UploadSheet_blur.csv', index=True, header=False) #存到csv
                    print("wrote")
            elif(labeltype[index[4]]=="不良-機械傷害"):
                print("不良-機械傷害")
                if(write_csv.loc[file_name][2]=="0"):
                    write_csv.loc[file_name][2]=1
                    write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\sigmoid Test_UploadSheet_blur.csv', index=True, header=False) #存到csv
                    print("wrote")
            elif(labeltype[index[4]]=="不良-炭疽病"):
                print("不良-炭疽病")
                if(write_csv.loc[file_name][3]=="0"):
                    write_csv.loc[file_name][3]=1
                    write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\sigmoid Test_UploadSheet_blur.csv', index=True, header=False) #存到csv
                    print("wrote")
            elif(labeltype[index[4]]=="不良-著色不佳"):
                print("不良-著色不佳")
                if(write_csv.loc[file_name][4]=="0"):
                    write_csv.loc[file_name][4]=1
                    write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\sigmoid Test_UploadSheet_blur.csv', index=True, header=False) #存到csv
                    print("wrote")
            elif(labeltype[index[4]]=="不良-黑斑病"):
                print("不良-黑斑病")
                if(write_csv.loc[file_name][5]=="0"):
                    write_csv.loc[file_name][5]=1   #指定 D5為 1
                    write_csv.to_csv('D:\\NTNU\\Pattern recognition\\final project\\Third\\sigmoid Test_UploadSheet_blur.csv', index=True, header=False) #存到csv
                    print("wrote")
        except Exception:
            pass
def ccPredict():
    #讀取csv資料
    file_path = 'D:\\NTNU\\Pattern recognition\\final project\\Second\\test_dev\\'          
    f_names = glob.glob(file_path + '*.jpg')                 #class'list'
    #label=['不良-乳汁吸附','不良-機械傷害','不良-炭疽病','不良-著色不佳','不良-黑斑病']
    label=['不良-著色不佳','不良-乳汁吸附','不良-炭疽病','不良-黑斑病','不良-機械傷害']
    
    train = np.genfromtxt('D:\\NTNU\\Pattern recognition\\final project\\Second\\dev_div.csv', delimiter=',', dtype=None)[1:]
    trainlabel = np.array([image_id for label,image_id in train]).astype(np.str)      
    #for i in range(len(trainlabel)):
        #print(trainlabel[i])
    pre=[]

    #讀取訓練完的model
    model = tensorflow.keras.models.load_model('D:\\NTNU\\Pattern recognition\\final project\\second\\mango90_model.h5')
    #j=0
    for i in range(len(f_names)):
        images = image.load_img(f_names[i], target_size=(90,90,3))
        x = image.img_to_array(images)
        
        # Standardize feature data
        x = x / 255
        x = np.expand_dims(x, 0)
        print('loading no.%s image' % i)
        
        #進行模型預測
        y = model.predict(x)
        print(y)
        index=np.argsort(y[0,:])     #排序
        print(label[index[4]],y[0,index[4]])          #("5",label[index[4]],y[0,index[4]])
        pre.append(label[index[4]])      #選最後的值(最大值)
        '''
        for file_name in os.listdir(filename_path):
            #data.append(file_name)
            with open('./final project/Second/dev record.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if(writer.writerow([(file_name[j]),(label[index[4]])])==0):
                    writer.writerow([(file_name[j]),(label[index[4]])])
            j=j+1
        #a=a+5
        #d+=1
        '''
    count=0

    for i in range(len(pre)):
        if pre[i]==trainlabel[i]:
            count=count+1
        
    print('Correct:',count)
    print('Accuracy:',count/len(f_names))
    
    pp=pd.crosstab(np.array(trainlabel), np.array(pre), rownames=['label'], colnames=['predict'])
    
    print(pp)
    print('Precision:',precision_score(trainlabel, pre, average='weighted'))
    print('Recall:',recall_score(trainlabel, pre, average='weighted'))
    print('f1_score:',f1_score(trainlabel, pre, average='weighted'))
    
#ccPredict()
#excel()
excelandpredict()
img=cv2.imread("D:/NTNU/Pattern recognition/final project/Third/Test/00015.jpg")
path="D:/NTNU/Pattern recognition/final project/Third/Test/"
label="不良-著色不佳"
bond = pd.read_csv('D:\\NTNU\\Pattern recognition\\final project\\Second\\Dev.csv' ,header=None,index_col=0)
#predict(img,"00015.jpg",path,bond)
tEnd=time.time()
print("It cost %f sec" % (tEnd - tStart))                       #會自動做進位,tEnd - tStart原型