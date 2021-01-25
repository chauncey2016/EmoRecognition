__author__ = 'ASUS'

import datetime
startTime = datetime.datetime.now()

basePath_Data = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/"              #local
# basePath_Data = "/nas/student/QiangChang/AffectNet/"                                                                      #nas
dataPath = basePath_Data + "FAUs/"                              #facial action units
# dataPath = basePath_Data + "RAW/"                                #landmarks

import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC

C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
kernel = ['linear','poly', 'rbf', 'sigmoid']  # 'precomputed'
recalls = np.zeros((len(C),len(kernel)))
accuracy = np.zeros((len(C),len(kernel)))

#Set SVM classifier and evaluate the model using accuracy, precision and Recall
def classifier(X_train,y_train,X_test,y_test):
    for i, iC in enumerate(C):
        for j, jK in enumerate(kernel):
            svm_clf = SVC(C=iC, kernel=jK,gamma='scale')
            #svm_clf = SVC(kernel='precomputed',gamma='scale')
            # if (jK == 'precomputed'):
            #     gram_train = np.dot(X_train, X_train.T)
            #     svm_clf.fit(gram_train, y_train)
            #     gram_test = np.dot(X_test, X_train.T)
            #     y_pred = svm_clf.predict(gram_test)
            #     y_pred_train = svm_clf.predict(gram_train)
            # else:
            svm_clf.fit(X_train,y_train)
            y_pred = svm_clf.predict(X_test)
            # y_pred_train = svm_clf.predict(X_train)
            #print(y_test)
            #print(y_pred)
            #Evaluating the Model
            #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            #print("Precision:",metrics.precision_score(y_test, y_pred))
            #print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
            #recalls[i][j] = metrics.recall_score(y_test, y_pred,average='micro')
            accuracy[i][j] = metrics.accuracy_score(y_test, y_pred)
            recalls[i][j] = metrics.recall_score(y_test, y_pred,average='macro')
            print("Recall of measuring on test data:" )
            df1 = pd.DataFrame(data=recalls,index=C,columns=kernel)
            print(df1)
            print("Acuracy of measuring on test data:" )
            df2 = pd.DataFrame(data=accuracy,index=C,columns=kernel)
            print(df2)
    return accuracy, recalls

Emo_labels = {0:'Neutral',1:'Happiness',2:'Sadness',3:'Surprise',4:'Fear',5:'Disgust',6:'Anger',7:'Contempt'}

#Do loops over instances, discard irrelevant classes in the dataset
def discard_irrel_cls(db):
    labels = pd.DataFrame(db).values[:,-3]
    for i,label in enumerate(labels):
        # if label == 'undefined':                        ###arousal and valence classification
        if label not in Emo_labels.keys():              ###Emotion classification
            db = db.drop([i])
    return db


def getData(file):
    '''
    get features and labels of the face images from the original csv file, which contains facial landmarks and labels information
    :param file: the location or path of the original csv file, which contains facial landmarks and labels information
    :return: features and corresponded labels
    '''

    df = pd.read_csv(file, sep=',')

    #coordinate of upperleft point of the face image
    loc_x = pd.DataFrame(df).values[:,1]
    loc_y = pd.DataFrame(df).values[:,2]
    #width and height of the face image
    width = pd.DataFrame(df).values[:,3]
    height = pd.DataFrame(df).values[:,4]
    #landmarks and labels of the face image
    pixels = pd.DataFrame(df).values[:,5]
    labels = pd.DataFrame(df).values[:,-3]

    features = []
    for i in range(len(df)):
        arr = [float(x) for x in pixels[i].split(";")]                 #the coordinates of landmarks are separated by semicolon
        landmarks = np.array(arr).reshape(68,2) #.swapaxes(1,0)
        ##landmarks normalisation
        for landmark in landmarks:
            landmark[0] = (landmark[0]-loc_x[i])/width[i]
            landmark[1] = (landmark[1]-loc_y[i])/height[i]
        features.append(landmarks.flatten())
    features = np.array(features)
    return features, labels


trainDataFile = "training.csv"
testDataFile = "validation.csv"

#dataset with discard operation
# trainDataFile = "training_with_discard.csv"
# testDataFile = "validation_with_discard.csv"

db_train = pd.read_csv(dataPath + trainDataFile, sep=';', header=None)   ##Using only header option, will either make header as data(if there exits heraders) or one of the data as header. So, better to use it with skiprows(skiprows = 1), this will create default header (1,2,3,4..) and remove the actual header of file.
# db_train = discard_irrel_cls(db_train)
df_train_data = pd.DataFrame(db_train).values[:,1:-3].astype(float)      ##array of features of train data
df_train_labels = pd.DataFrame(db_train).values[:,-3].astype(float)      ##array of labels of train data
# df_train_data, df_train_labels = getData(dataPath + trainDataFile)

db_test = pd.read_csv(dataPath + testDataFile, sep=';', header=None)
# db_test = discard_irrel_cls(db_test)
df_test_data = pd.DataFrame(db_test).values[:,1:-3].astype(float)      ##array of features of test data
df_test_labels = pd.DataFrame(db_test).values[:,-3].astype(float)      ##array of labels of test data
# df_test_data, df_test_labels = getData(dataPath + testDataFile)

# # count the number of instances of each class
# count = 0
# for label in df_train_labels:
# # for label in df_test_labels:
#     if label == 10:
#         count = count+1
#
# print(count)

train = list(set(df_train_labels))
test = list(set(df_test_labels))
print('-------  Label Information  -------')
print("The class label for train data: " + str(train))
print("The class label for  test data: " + str(test))

#Standardize the features using sklearn
scaler = preprocessing.StandardScaler().fit(df_train_data)
train_features = scaler.transform(df_train_data)
test_features = scaler.transform(df_test_data)

#Convert labels into classes--------------------------------------------
classConverter = preprocessing.LabelEncoder()
classConverter.fit(df_train_labels)
train_labels_classes = classConverter.transform(df_train_labels)
test_labels_classes = classConverter.transform(df_test_labels)

UARs = np.zeros((len(C),len(kernel)))
accs = np.zeros((len(C),len(kernel)))
accs, UARs = classifier(df_train_data,train_labels_classes,df_test_data,test_labels_classes)
#-------------------------------------------------------------------------
# svm_clf = SVC(C=0.01, kernel='linear',gamma='scale')
# svm_clf.fit(train_features,train_labels_classes)
# y_pred = svm_clf.predict(test_features)
# recall = metrics.recall_score(test_labels_classes, y_pred,average='micro')

print('-----------------Evalute information------------------')
print('The accuracy of SVM model with Landmarks: ' )
df_acc = pd.DataFrame(data=accs,index=C,columns=kernel)
print(df_acc)

print('The UARs of SVM model with Landmarks: ' )
df_uar = pd.DataFrame(data=UARs,index=C,columns=kernel)
print(df_uar)
endTime = datetime.datetime.now()
duration  = endTime - startTime
seconds = duration.total_seconds()
hours = seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print("The program has run " + '{} hours, {} minutes, {} seconds'.format(hours, minutes, seconds))



######### 1.FAUs or VGGface?   2.which parts of data in the csv file are seen as features and labels?

# on val dataset
#    task                         SVC             accuracy            time
# Emotion(FAUs)                linear/0.1         23.3%             69h 46min