#data standardization : StandardScaler/ MinMaxScaler ,the same scaler used by differnet model(svm or nn) may has completely different effect


#Fine_tune:   1.Discard irrelevant classes    2.the parameter of metrics.recall_score(): average= from 'micro' to 'macro'

#preparing data path
basePath_Data = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/"

import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
kernel = ['linear','poly', 'rbf', 'sigmoid', 'precomputed']
accs = np.zeros((len(C),len(kernel)))
recalls = np.zeros((len(C),len(kernel)))

#Set SVM classifier and evaluate the model using accuracy, precision and Recall
def classifier(X_train,y_train,X_test,y_test):
    for i, iC in enumerate(C):
        for j, jK in enumerate(kernel):
            svm_clf = SVC(C=iC, kernel=jK, gamma='scale')
            #svm_clf = SVC(kernel='precomputed',gamma='scale')
            if (jK == 'precomputed'):                            ############ when kernel='precomputed',we should then pass Gram matrix instead of X to the fit and predict methods
                gram_train = np.dot(X_train, X_train.T)
                svm_clf.fit(gram_train, y_train)
                gram_test = np.dot(X_test, X_train.T)
                y_pred = svm_clf.predict(gram_test)
            else:
                svm_clf.fit(X_train,y_train)
                y_pred = svm_clf.predict(X_test)
            #Evaluating the Model
            #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            #print("Precision:",metrics.precision_score(y_test, y_pred))
            #print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
            accs[i][j] = metrics.accuracy_score(y_test, y_pred)
            recalls[i][j] = metrics.recall_score(y_test, y_pred,average='macro')    #### every combination of C and kernel has a corresponded recall

    return accs, recalls



#data files(feutres of lld and functional)
# train_file = "EmoDB_training_.csv"  #feature data used for training the model
# devel_file = "EmoDB_devel_.csv"
# test_file = "EmoDB_test_.csv"

#data files(feutres of lld)
# train_file = "EmoDB_lld_training.csv"  #feature data used for training the model
# devel_file = "EmoDB_lld_devel.csv"
# test_file = "EmoDB_lld_test.csv"

train_file = "ComParE_training.csv"  #feature data used for training the model
devel_file = "ComParE_devel.csv"
test_file = "ComParE_test.csv"

label_file = "lab/ComParE2013_Emotion.tsv"


#Read and prepare the data
db_train = pd.read_csv(basePath_Data + train_file, sep=';')
train_data = pd.DataFrame(db_train).values[:,2:]      ##array of features data for training

db_devel = pd.read_csv(basePath_Data + devel_file, sep = ';')
devel_data = pd.DataFrame(db_devel).values[:,2:]

db_test = pd.read_csv(basePath_Data + test_file, sep = ';')
test_data = pd.DataFrame(db_test).values[:,2:]

#read and prepare corresponded labels
db_labels = pd.read_csv(basePath_Data + label_file,sep="\t")
train_labels = pd.DataFrame(db_labels).values[216:,-2]
devel_labels = pd.DataFrame(db_labels).values[0:216,-2]


def discard_irrel_cls(df_data, df_labels):
    for i,label in enumerate(df_labels.values):
        if label == 'undefined': #'other' for emotion classification, 'undefined' for arousal and valence classification
            df_data = df_data.drop([i])
            df_labels = df_labels.drop([i])
    return df_data.values, df_labels.values

# print(type(db_labels), db_labels.shape)
train_data, train_labels = discard_irrel_cls(pd.DataFrame(train_data), pd.DataFrame(train_labels))
devel_data, devel_labels = discard_irrel_cls(pd.DataFrame(devel_data), pd.DataFrame(devel_labels))

# train = list(set(train_labels))
# test = list(set(devel_labels))
# print('-------  Label Information  -------')
# print("The class label for train data: " + str(train))
# print("The class label for  test data: " + str(test))
# for j,speaker_id in enumerate(indexes_speaker):
#     dataTrain = []
#     train_labels = []
#     dataTest = []
#     test_labels = []
#     for i in range(len(df_names)):           #loops over records
#         if speaker_id in df_names[i][1:3]:   #e.g.   03(speaker_id) and '03a01Fa'(df_names[i]), "'" is also a part of the string(df_names[i])
#             dataTest.append(df_data[i])
#             test_labels.append(df_names[i][-3])
#         else:
#             dataTrain.append(df_data[i])
#             train_labels.append(df_names[i][-3])
#
#     train = list(set(train_labels))
#     test = list(set(test_labels))
#     print(train,test)
#
#     print("Use speaker_" + speaker_id + " as test data and others as train data to train the model")

#Standardize the features using sklearn
# scaler = preprocessing.StandardScaler().fit(train_data)                            #??????????????? here can I respectively fit train_feature and test_feature to get the Scaler or can I just fit one of them?
# train_features = scaler.transform(train_data)
# devel_features = scaler.transform(devel_data)
scaler = preprocessing.StandardScaler()
train_features = scaler.fit_transform(train_data)
devel_features = scaler.fit_transform(devel_data)

#Convert labels into classes
classConverter = preprocessing.LabelEncoder()
#classConverter.fit(np.concatenate((train_labels,test_labels)))
classConverter.fit(train_labels)                                                            #  #???????????????????????? the same question like upside
train_labels_classes = classConverter.transform(train_labels)
devel_labels_classes = classConverter.transform(devel_labels)


print("The shape of train and devel data: {},{}".format(train_features.shape, devel_features.shape))

accuracies, UARs = classifier(train_features,train_labels_classes,devel_features,devel_labels_classes)

print('-----------------Evalute information------------------')
print("Accuracy:")
df_acc = pd.DataFrame(data=accuracies,index=C,columns=kernel)
print(df_acc)
print("Recall:")
df_uar = pd.DataFrame(data=UARs,index=C,columns=kernel)
print(df_uar)

print(" Mean  value of recall: {}".format(np.mean(UARs,axis=0)))      ##### when axis=0 ,compute mean of each column of the array, when axis=1,compute mean of each row of the array. The default is to compute the mean of the flattened array.
print("Median value of recall: {}".format(np.median(UARs,axis=0)))










#Set SVM classifier and check the predictions with cross validation
#classifier = SVC(C = 1e-2, kernel = 'linear')
#predictions = cross_val_predict(classifier, train_features, train_labels_classes, cv=10)

#report = classification_report(train_labels_classes, predictions, target_names = classConverter.classes_)
#print(report)

#accuracy = accuracy_score(train_labels_classes, predictions)
#print('Accuracy: ' + str(accuracy*100) + ' %')

#from sklearn.metrics import recall_score
#UAR = recall_score(train_labels_classes, predictions, average='macro')
#print(UAR)


#Fine-tune parameters
#import numpy as np
#import matplotlib.pyplot as plt
#C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
accuracies = np.zeros_like(C)
def fine_tune():
    for i, iC in enumerate(C):
        classifier = SVC(C = iC, kernel = 'linear')
        predictions = metrics.cross_val_predict(classifier, train_features, train_labels_classes, cv=10)
        accuracies[i] = metrics.accuracy_score(train_labels_classes, predictions)

    plt.plot(np.arange(len(C)), 100*accuracies)
    plt.grid()
    plt.xlabel('C parameter')
    plt.ylabel('Accuracy (%)')
    plt.xticks(np.arange(len(C)), C)
    plt.show()

#fine_tune()

#Test SVC with the optimal parameters
from sklearn.metrics import classification_report

# db_test = pd.read_csv(basePath_Data + test_file, sep = ';')
# test_data = pd.DataFrame(db_test).values[:,1:]
#
# test_features = scaler.transform(test_data)

def test(train_features,train_labels_classes,test_features,test_labels_classes ):
    # classifier = SVC(C = 1, kernel = 'sigmoid') # 选择上一步plt图中最准确的参数    Emotion-eGeMAPs
    classifier = SVC(C = 1, kernel = 'linear') # 选择上一步plt图中最准确的参数    Emotion-eGeMAPs
    # classifier = SVC(C = 0.01, kernel = 'linear')  # Arousal
    # classifier = SVC(C = 0.1, kernel = 'linear')  # Valence

    classifier.fit(train_features, train_labels_classes)
    predictions = classifier.predict(test_features)

    # report = classification_report(test_labels_classes, predictions,  target_names = classConverter.classes_)
    # print(report)

    accuracy = metrics.accuracy_score(test_labels_classes, predictions)
    print('Accuracy: ' + str(accuracy*100) + ' %')

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(test_labels_classes, predictions))

# test(train_features,train_labels_classes,devel_features,devel_labels_classes)




# On test data
#    task                                       SVC             accuracy
# Emotion classification(eGeMAPS)           linear/0.01         34.3750%
# Emotion classification(eGeMAPS)           sigmoid/1.0         34.8958%   ------------------
# Emotion classification(IS13_ComParE)      sigmoid/1.0           34%
# Emotion classification(IS13_ComParE)      linear/0.1          35.9375%  --------------------
# Arousal(eGeMAPS)                          linear/0.01         81.25%  ----
# Valence(eGeMAPS)                          linear/0.1          69.71%  ----
# Arousal(IS13_ComParE)                     linear/0.001        85.10%  ----
# Valence(IS13_ComParE)                     linear/1.0          77.88%  ----

