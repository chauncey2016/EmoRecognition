__author__ = 'ASUS'

import datetime
import os
import csv
from scipy.io import arff
from pandas import read_csv
from pandas import DataFrame
from numpy import zeros, mean, asarray
from numpy import vstack
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

import torch
from torch import tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU, Softmax, Sigmoid
from torch.nn import Sequential
from torch.nn import Conv1d,Conv2d
from torch.nn import BatchNorm1d,BatchNorm2d
from torch.nn import MaxPool1d,MaxPool2d
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


EPOCHS = 300
Leaning_Rate = 0.001
BATCH_SIZE = 32
# Data_Path = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/"
Data_Path ="/nas/student/QiangChang/ComParE2013_Emotion/dist/"

torch.manual_seed(0)

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, data_path, label_path):
        # load the csv file as a dataframe
        df_f = read_csv(data_path, sep=';')                #lld and funtional features--------------------------------------
        # df_f = read_csv(data_path, sep=',', header=None)    #lld
        if os.path.splitext(label_path)[1] == '.csv':
            df_l = read_csv(label_path)
            labels = df_l.values[:, 1]
        else:
            df_l = read_csv(label_path, sep='\t')
            labels = df_l.values[:, -3]
        # store the inputs and outputs
        self.fname = df_f.values[:, 0]
        self.lname = df_l.values[:, 0]
        self.X = df_f.values[:, 2:]     #lld and funtional features-------------------------------------------------------
        # self.X = df_f.values[:,1:]    #lld features
        # if 'training' in os.path.basename(data_path): ##the first 216 rows are devel labes,the others are train labels
        #     self.y = df_l.values[216:, 3]
        # else:
        #     self.y = df_l.values[0:216, 3]
        self.y = []
        for i, iname in enumerate(self.fname):
            for j, jname in enumerate(self.lname):
                if iname.strip('\'') == jname.strip('\''):
                    self.y.append(labels[j])
                    break
        #discard irrelevant classes in the dataset
        self.X, self.y = self.discard_irrel_cls(DataFrame(self.X), DataFrame(self.y))
        #Standardize the features using sklearn
        # scaler = StandardScaler().fit(self.X)
        scaler = MinMaxScaler().fit(self.X)                            #??????????????? here can I respectively fit train_feature and test_feature to get the Scaler or can I just fit one of them?
        self.X = scaler.transform(self.X)
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # self.X = self.X.reshape(self.X.shape[0],1,88)   #lld and funtional features with eGeMAPS--------------------------------------------------------------
        self.X = self.X.reshape(self.X.shape[0],1,6373) #lld and funtional features with IS13_ComParE--------------------------------------------------------------
        # self.X = self.X.reshape(self.X.shape[0],1,248,23)   #lld features
        # label encode target and ensure the values are floats
        if len(self.y):
            self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')          ##used for arousal and valence----------------------------------------------------------------
        self.y = self.y.reshape((len(self.y), 1))   ##used for arousal and valence

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    #Do loops over instances, discard irrelevant classes in the dataset
    def discard_irrel_cls(self, df_data, df_labels):
        for i, label in enumerate(df_labels.values):
            #print(i, label)
            if label == 'undefined':            #'undefined' for arousal and valence
                # self.X = delete(original_X, i, axis = 0)
                # self.y = delete(original_y, i, axis = 0)
                df_data = df_data.drop([i])
                df_labels = df_labels.drop([i])
        return df_data.values, df_labels.values


class ALDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, sep=';', header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.X = self.X.reshape(self.X.shape[0],1,1,6373)
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition( Multilayer Perceptrons (MLP) )
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 12)  #12
        self.activation = Softmax(dim=1)
        # self.activation = Sigmoid()          ##just for BCELoss; when using CrosEntropyLoss, don't need the activation function of last layer

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X

class MLP3(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP3, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 3000)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(3000, 1000)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer
        self.hidden3 = Linear(1000, 200)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # fourth hidden layer and output
        self.hidden4 = Linear(200, 1)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)
        # self.act4 = Sigmoid()       ##just for BCELoss; when using CrosEntropyLoss, don't need the activation function of last layer

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        # X = self.act4(X)
        return X

class CNND(Module):
    def __init__(self):
        super(CNND, self).__init__()
        self.layer1 = Sequential(
            Conv1d(1, 3, kernel_size=5, stride=1),  ####, padding=1
            BatchNorm1d(3),
            ReLU(inplace=True),
            MaxPool1d(3))
        self.layer2 = Sequential(
            Conv1d(3, 9, kernel_size=5, stride=1),
            BatchNorm1d(9),
            ReLU(inplace=True),
            MaxPool1d(3))
        self.layer3 = Sequential(
            Conv1d(9, 27, kernel_size=5, stride=1),
            BatchNorm1d(27),
            ReLU(inplace=True),
            MaxPool1d(3))
        # self.fc = Linear(9, 12)
        self.fc = Sequential(
            Linear(6318, 1024),
            ReLU(inplace=True),
            Linear(1024, 256),
            ReLU(inplace=True),
            Linear(256, 64),
            ReLU(inplace=True),
            Linear(64, 12)
            # Softmax()
            # Sigmoid()           ##just for BCELoss; when using CrosEntropyLoss, don't need the activation function of last layer
        )

    def forward(self, x):
        out = self.layer1(x)
        # print('first layer output {}'.format(out.shape))
        out = self.layer2(out)
        # print('second layer output {}'.format(out.shape))
        out = self.layer3(out)
        # print('third layer output {}'.format(out.shape))
        out = out.view(out.size(0), -1)
        # print('third1 layer output {}'.format(out.shape))
        out = self.fc(out)
        return out

class CNN1D(Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = Sequential(
            Conv1d(1, 8, kernel_size=5, stride=1),  ####, padding=1
            BatchNorm1d(8),
            ReLU(inplace=True),
            MaxPool1d(4))
        self.layer2 = Sequential(
            Conv1d(8, 16, kernel_size=5, stride=1),
            BatchNorm1d(16),
            ReLU(inplace=True),
            MaxPool1d(4))
        self.layer3 = Sequential(
            Conv1d(16, 64, kernel_size=5, stride=1),
            BatchNorm1d(64),
            ReLU(inplace=True),
            MaxPool1d(4))
        # self.fc = Linear(10*1, 12)
        self.fc = Sequential(
            # Linear(7*11*1, 4096),
            Linear(6272, 1024),
            Linear(1024, 256),
            # ReLU(inplace=True),
            Linear(256, 32),
            # ReLU(inplace=True),
            Linear(32, 12)
            # Sigmoid()             ##just for BCELoss; when using CrosEntropyLoss, don't need the activation function of last layer
        )

    def forward(self, x):
        out = self.layer1(x)
        # print('first layer output {}'.format(out.shape))
        out = self.layer2(out)
        # print('second layer output {}'.format(out.shape))
        out = self.layer3(out)
        # print('third layer output {}'.format(out.shape))
        out = out.view(out.size(0), -1)
        # print('third1 layer output {}'.format(out.shape))
        out = self.fc(out)
        return out

class CNN2D(Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 8, kernel_size=(1,5), padding=0),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d((1,4)))
        self.layer2 = Sequential(
            Conv2d(8, 16, kernel_size=(1,5), padding=0),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d((1,3)))
        self.layer3 = Sequential(
            Conv2d(16, 64, kernel_size=(1,5), padding=0),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d((1,2)))
        # self.fc = Linear(5*22*1, 12)
        self.fc = Sequential(
            Linear(262*64, 4096),
            ReLU(inplace=True),
            Linear(4096, 1024),
            ReLU(inplace=True),
            Linear(1024, 256),
            ReLU(inplace=True),
            Linear(256, 12)
        )

    def forward(self, x):
        out = self.layer1(x)
        # print('first layer output {}'.format(out.shape))
        out = self.layer2(out)
        # print('second layer output {}'.format(out.shape))
        out = self.layer3(out)
        # print('third layer output {}'.format(out.shape))
        out = out.view(out.size(0), -1)
        # print('third1 layer output {}'.format(out.shape))
        out = self.fc(out)
        return out

# train the model
def train_model(train_dl, model, device, epochs):
    # define the optimization----------------------------------------------------------------------------------------------
    criterion = CrossEntropyLoss()
    # criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=Leaning_Rate, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=0.01)
    # enumerate epochs
    for epoch in range(epochs):
        myloss = 0.
        counter = 0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            counter += 1
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # # label encode target and ensure the values are floats
            targets = targets.long()
            # calculate loss
            loss = criterion(yhat, targets)
            myloss += loss.detach().cpu().numpy()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        print('The mean loss of epoch ' + str(epoch) + ' is: %.3f' % (myloss/counter))

# evaluate the model
def evaluate_model1(test_dl, model, device):
    model.eval()
    with torch.no_grad():
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            inputs = inputs.to(device)
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().cpu().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        print(confusion_matrix(actuals, predictions))
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        rec = recall_score(actuals, predictions, average='macro')
    return acc,rec

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

def evaluate(test_dl, model, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = outputs.data.ge(0.5)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
    print('Test Accuracy of the model on the test data: %.4f %%' % (100 * correct / total))

def active_learning(dataloader, model, device):

    update_AL_set = list()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        # predict with the active learning data set
        yhat = model(inputs)#softmax
        act = Softmax()
        yhat = act(yhat)
        # confidence score
        yhat_prob, yhat_cls = torch.max(yhat, axis=1) #probability and label

        # retrieve numpy array
        yhat_prob = yhat_prob.detach().cpu().numpy()
        yhat_cls = yhat_cls.detach().cpu().numpy()
        # print("Prob of Batch {}: {}".format(i+1, yhat_prob))
        # yhat_cls = argmax(yhat, axis=1)
        # print("Cls1 of Batch {}: {}".format(i+1, yhat_cls))
        # expert annotation
        actual = targets.numpy()
        # sample selection function: If the confidence score is blew the threshold, ask the annotator to manually annotate that sample (in practice, just take the annotation from this sample directly from the dataset)
        for j in range(len(yhat)):
            if yhat_prob[j] < 0.1:
                yhat_cls[j] = actual[j]
            new_input = inputs[j].squeeze().tolist()
            new_input.extend([yhat_cls[j]])
            update_AL_set.append(new_input)
        # print("Cls2 of Batch {}: {}".format(i+1, yhat_cls))
    # print(update_AL_set)
    csv.register_dialect('mydialect', delimiter=';')
    al_dataset_csv = Data_Path + 'AL_dataset.csv'
    if os.path.exists(al_dataset_csv):
        os.remove(al_dataset_csv)
    with open(al_dataset_csv,'w', newline='') as csvfile:
        writer = csv.writer(csvfile, 'mydialect')
        for row in update_AL_set:
            writer.writerow(row)

    AL_dataset = ALDataset(al_dataset_csv)
    AL_dl =  DataLoader(AL_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    return  AL_dl


def main():
    # time
    print("Started time:{}".format(datetime.datetime.now()))
    # start to train the model
    # trainFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/EmoDB_lld_training.csv"  #lld feature data used for training the model
    # develFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/EmoDB_lld_devel.csv"
    # labelFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/lab/ComParE2013_Emotion.tsv"
    # trainFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/EmoDB_training_.csv"  #feature data used for training the model
    # develFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/EmoDB_devel_.csv"
    # labelFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/lab/ComParE2013_Emotion.tsv"
    # trainFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/ComParE_training.csv"  #feature data used for training the model
    # trainFileA = Data_Path + "ComParE_training_A.csv"
    # trainFileB = Data_Path + "ComParE_training_B.csv"
    trainFile = Data_Path + "ComParE_training.csv"
    develFile = Data_Path + "ComParE_devel.csv"
    testFile = Data_Path + "ComParE_test.csv"
    labelFile = Data_Path + "lab/ComParE2013_Emotion.tsv"
    test_labelFile = Data_Path + "lab/ComParE2013_Emotion.test.lab.csv"
    # trainFile = '/nas/student/QiangChang/ComParE2013_Emotion/dist/EmoDB_lld_training.csv'
    # develFile = '/nas/student/QiangChang/ComParE2013_Emotion/dist/EmoDB_lld_devel.csv'
    # labelFile = '/nas/student/QiangChang/ComParE2013_Emotion/dist/lab/ComParE2013_Emotion.tsv'
    # trainFileA = '/nas/student/QiangChang/ComParE2013_Emotion/dist/ComParE_training_A.csv'
    # trainFileB = '/nas/student/QiangChang/ComParE2013_Emotion/dist/ComParE_training_B.csv'
    # develFile = '/nas/student/QiangChang/ComParE2013_Emotion/dist/ComParE_devel.csv'
    # labelFile = '/nas/student/QiangChang/ComParE2013_Emotion/dist/lab/ComParE2013_Emotion.tsv'


    # train_data_A = CSVDataset(trainFileA,labelFile)
    # train_data_B = CSVDataset(trainFileB,labelFile)
    train_data = CSVDataset(trainFile,labelFile)
    devel_data = CSVDataset(develFile,labelFile)
    test_data = CSVDataset(testFile,test_labelFile)
    # prepare data loaders
    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    # al_dl = DataLoader(train_data_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    devel_dl = DataLoader(devel_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    test_dl = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    print('-------  Data Information  -------')
    print("The size of train, devel and test data: {},{},{}" .format(len(train_dl.dataset), len(devel_dl.dataset), len(test_dl.dataset)))
    print("The shape of train and devel data: {},{}" .format(train_data.X.shape, devel_data.X.shape))
    print("The shape of train and devel labels: {},{}" .format(train_data.y.shape, type(devel_data.y)))
    print("The shape of test data and labels: {},{}" .format(test_data.X.shape, test_data.y.shape))
    # print(train_data_A.__getitem__(0))

    # train = list(set(train_data_A.y))
    # # train = train_data_A.y.unique()
    # devel = list(set(devel_data.y))
    # print('-------  Label Information  -------')
    # print("The class label for train data: " + str(train))
    # print("The class label for  test data: " + str(devel))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the network ------------------------------------------------------------------------------------------------------------------------------------
    # model = CNN2D().to(device)
    model = MLP3(6373).to(device)    #should be 88

    # train the model
    print("----------------------First training------------------------")
    train_model(train_dl, model, device, EPOCHS)
    # evaluate the model-------------------------------------------------------------------------------------------------------------------------------------
    accs, recall = evaluate_model1(devel_dl, model, device)
    # accs = evaluate_model(devel_dl, model)

    print("Evaluating the Model:{} with Epochs={}, lr={} and Batch_size={}".format(model._get_name(), EPOCHS, Leaning_Rate, BATCH_SIZE))
    print('UAR: %.3f' % recall)
    print('Accuracy: %.3f' % accs)


    # # train the optimal model using data belonging to the Subset A of the training data and the development data
    # list_of_datasets = []
    # list_of_datasets.append(train_data_A)
    # list_of_datasets.append(devel_data)
    # TD_dataset = ConcatDataset(list_of_datasets)
    # # print(TD_dataset.__len__())
    #
    # train_dl = DataLoader(TD_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    #
    # print("----------------------Second training------------------------")
    # train_model(train_dl, model, device, 100)
    #
    # evaluate(test_dl, model, device)
    #
    # # active learning
    # AL_dl = active_learning(al_dl, model, device)
    # # re-train the model
    # print("----------------------Third training(AL)------------------------")
    # train_model(AL_dl, model, device, 50)
    # # assess the model performance
    # evaluate(test_dl, model, device)


if __name__ == "__main__":
    main()


####-------Questions---------------
# 1. How can I usually define the nn? How many layers? How many perceptrons in the hidden layer?
# 2. Potential Problems:   Data preparing/ data preprocessing(transformation/too much input features, Remove or?)/ model structure




#fine_tune:
#1.pytorch loss fuction: CrossEntropyLoss() has already assigned some activation fuction of the output layer for the network, so remove the act() of the output layer
#2.data standardization: MinmaxScaler()




# Evaluating
# Model     Epochs     Lr        Batch_size       acc              Task
# MLP3       360       001         32             27.1%           Emotion
# MLP3       380       001         32             29.2%           Emotion   ------------------------
# MLP3       390       001         32             25%             Emotion
# MLP3       380       001         16             26%             Emotion
# MLP3       300       0001        32             39.1%(38.8)     Emotion(IS13_ComParE) ---------------------
# MLP3       350       0005        32             38.5%(42.5)     Emotion(IS13_ComParE)
# CNN1D       10       005         48             24.2%           Emotion  -----------------------------
# CNN1D      100       0005        4              21.4%           Emotion
# CNND       200       0001        32             32.3%(33.1)     Emotion(IS13_ComParE) -----------------
# CNN2D      100       0001        32             32.8%(35.4)     Emotion(IS13_ComParE)  ----------------



# MLP3       600       0005        32             80.3%           Arousal
# MLP3       600       0001        32             80.8%           Arousal -------------
# CNN1d      50        0001        32             74.5%           Arousal --------------
# MLP3       500       0001        48             71.6%           Valence -------------
# MLP1       200       0001        32             71.2%           Valence
# CNN1d      20        0001        32             67.3%           Valence --------------



# Active learning
# Model     Lr        Batch_size               Task             Threshold            Epoch1    Acc1      Epoch2    Acc2     Epoch3    Acc3
# CNN2D    0001        32             Emotion(IS13_ComParE)        0.6               100       33.9%      100       34%      50       36%
# CNN2D    0001        32             Emotion(IS13_ComParE)        0.2               100       33.3%      100       33%      50       37%
# CNN2D    0001        32             Emotion(IS13_ComParE)        0.1               100       33.9%      100       34%      50       36%