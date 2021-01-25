__author__ = 'ASUS'

import datetime
startTime = datetime.datetime.now()

import os
import io
import csv
from pandas import read_csv
from pandas import DataFrame
from numpy import array, asarray, delete
from numpy import zeros,mean
from numpy import vstack
from numpy import argmax
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import torch
from torch.nn import Linear
from torch.nn import ReLU, Dropout
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Sequential
from torch.nn import Conv1d, Conv2d
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import MaxPool1d, MaxPool2d
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torchvision import transforms

DATA_SET = "RAW IMAGEs"   #Landmarks   #FAUs    #RAW IMAGEs
Emo_labels = {0:'Neutral',1:'Happiness',2:'Sadness',3:'Surprise',4:'Fear',5:'Disgust',6:'Anger',7:'Contempt'}
EPOCHS = 40
Leaning_Rate = 0.001
BATCH_SIZE = 48

# Data_Path = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/FAUs/"
# Data_Path = '/nas/student/QiangChang/AffectNet/FAUs/'     #FAUs
Data_Path = '/nas/student/QiangChang/AffectNet/RAW/'        #RAW IMAGEs

#####################1d to 2d:  model; input dimension;

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, sep=',', header=None)
        # df = self.discard_irrel_cls(df)
        # store the inputs and outputs
        self.X = df.values[:,1:-3]                              #FAUs
        self.y = df.values[:,-3]
        # self.X, self.y = self.getData(path)                            #landmarks
        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.X = self.X.reshape(self.X.shape[0],1,17)           #FAUs and cnn
        # self.X = self.X.reshape(self.X.shape[0],1,68,2)         #landmarks and cnn
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    #Do loops over instances, discard irrelevant classes in the dataset
    def discard_irrel_cls(self, df):
        labels = DataFrame(df).values[:,-3]
        for i, label in enumerate(labels):
            #print(i, label)
            if label not in Emo_labels.keys():
                # self.X = delete(original_X, i, axis = 0)
                # self.y = delete(original_y, i, axis = 0)
                df = df.drop([i])
        return df

    # landmarks
    def getData(self, file):
        '''
        get features and labels of the face images from the original csv file, which contains facial landmarks and labels information
        :param file: the location or path of the original csv file, which contains facial landmarks and labels information
        :return: features and corresponded labels
        '''

        df = read_csv(file, sep=',')

        #coordinate of upperleft point of the face image
        loc_x = DataFrame(df).values[:,1]
        loc_y = DataFrame(df).values[:,2]
        #width and height of the face image
        width = DataFrame(df).values[:,3]
        height = DataFrame(df).values[:,4]
        #landmarks and labels of the face image
        pixels = DataFrame(df).values[:,5]
        labels = DataFrame(df).values[:,-3]

        features = []
        for i in range(len(df)):
            arr = [float(x) for x in pixels[i].split(";")]                 #the coordinates of landmarks are separated by semicolon
            landmarks = array(arr).reshape(68,2) #.swapaxes(1,0)
            ##landmarks normalisation
            for landmark in landmarks:
                landmark[0] = (landmark[0]-loc_x[i])/width[i]
                landmark[1] = (landmark[1]-loc_y[i])/height[i]
            # features.append(landmarks)    ##---------------------------------------------------------------------------------------------------------- CNN
            features.append(landmarks.flatten())   #when the model is MLP, flatten the input data as 1-dimensional
        features = array(features)
        return features, labels

class ImageDataset(Dataset):

    def __init__(self, csvfile):

        image_path = Data_Path + "Manually_Annotated_Images/"

        # df = read_csv(csvfile, sep=',')
        df = read_csv(csvfile, sep=',', header=None) ##irrelevant discard files
        self.file_names = DataFrame(df).values[:,0]
        img_bounding_box = DataFrame(df).values[:, 1:5]
        file_labels = DataFrame(df).values[:,-3]

        loadedImages = []
        for name in self.file_names:
            loadedImages.append(os.path.join(image_path, name))
        self.images = loadedImages
        self.bounding_box = img_bounding_box
        self.labels = LabelEncoder().fit_transform(file_labels)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
        ])

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        imgArray = asarray(img)
        face_loc_x = self.bounding_box[idx][0]
        face_loc_y = self.bounding_box[idx][1]
        face_width = self.bounding_box[idx][2]
        face_height = self.bounding_box[idx][3]
        faceArray = imgArray[face_loc_x:face_loc_x+face_width+1, face_loc_y:face_loc_y+face_height+1, ]
        faceTensor = self.transforms(faceArray)
        # #Normalise
        # max1 = torch.max(faceTensor,2)[0]
        # max = torch.max(max1,1)[0]
        # min1 = torch.min(faceTensor,2)[0]
        # min = torch.min(min1,1)[0]
        # neg_min = torch.mul(min, -1)
        # for i in range(faceTensor.size()[0]):
        #     faceTensor[i] = torch.add(faceTensor[i],neg_min[i].item())   #(px - min)
        #     max_min = torch.add(max[i],neg_min[i].item()).item()     #(max - min)
        #     faceTensor[i] = torch.div(faceTensor[i], max_min)      #(px - min)/(max - min)

        return [faceTensor, self.labels[idx]]

    def __len__(self):
        return len(self.images)

# model definition( Multilayer Perceptrons (MLP) )
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):                             # should  n_inputs equal the number of features of each example?
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 8)   # 11 or 8
        # self.activation = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        # X = self.activation(X)
        return X

class MLP1(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP1, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 68)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(68, 34)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer
        self.hidden3 = Linear(34, 17)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # fourth hidden layer and output
        self.hidden4 = Linear(17, 8)
        xavier_uniform_(self.hidden4.weight)
        # self.act3 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        # X = self.act4(X)
        return X


class CNN1D(Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = Sequential(
            Conv1d(1, 16, kernel_size=2, stride=1),  ####, padding=1
            BatchNorm1d(16),
            ReLU(inplace=True),
            MaxPool1d(2))
        self.layer2 = Sequential(
            Conv1d(16, 64, kernel_size=2, stride=1),
            BatchNorm1d(64),
            ReLU(inplace=True),
            MaxPool1d(2))
        self.layer3 = Sequential(
            Conv1d(64, 128, kernel_size=2, stride=1),
            BatchNorm1d(128),
            ReLU(inplace=True),
            MaxPool1d(2))
        self.fc = Linear(128, 8)

    def forward(self, x):
        out = self.layer1(x)
        # print('first layer output {}:{}'.format(out.shape, out))
        out = self.layer2(out)
        # print('second layer output {}:{}'.format(out.shape, out))
        out = self.layer3(out)
        # print('third layer output {}:{}'.format(out.shape, out))
        out = out.view(out.size(0), -1)
        # print('third1 layer output {}'.format(out.shape))
        out = self.fc(out)
        return out

class CNND(Module):
    def __init__(self):
        super(CNND, self).__init__()
        self.layer1 = Sequential(
            Conv1d(1, 8, kernel_size=2, stride=1),  ####, padding=1
            BatchNorm1d(8),
            ReLU(inplace=True),
            MaxPool1d(2))
        self.layer2 = Sequential(
            Conv1d(8, 16, kernel_size=2, stride=1),
            BatchNorm1d(16),
            ReLU(inplace=True),
            MaxPool1d(2))
        self.layer3 = Sequential(
            Conv1d(16, 64, kernel_size=2, stride=1),
            BatchNorm1d(64),
            ReLU(inplace=True),
            MaxPool1d(2))
        self.fc = Linear(64*1, 8)

    def forward(self, x):
        out = self.layer1(x)
        # print('first layer output {}:{}'.format(out.shape, out))
        out = self.layer2(out)
        # print('second layer output {}:{}'.format(out.shape, out))
        out = self.layer3(out)
        # print('third layer output {}:{}'.format(out.shape, out))
        out = out.view(out.size(0), -1)
        # print('third1 layer output {}'.format(out.shape))
        out = self.fc(out)
        return out

class CNN2D(Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.layer1 = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1),
            # Conv2d(1, 16, kernel_size=(1,3), stride=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            # MaxPool2d(1,2))
            MaxPool2d(2))
        self.layer2 = Sequential(
            Conv2d(16, 64, kernel_size=3, stride=1),
            # Conv2d(16, 64, kernel_size=(1,3), stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            # MaxPool2d(1,2))
            MaxPool2d(2))
        self.layer3 = Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1),
            # Conv2d(64, 128, kernel_size=(1,3), stride=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            # MaxPool2d(1,2))
            MaxPool2d(2))
        # self.fc = Linear(1*1*128, 8)
        self.fc = Sequential(
            Dropout(p=0.2),
            Linear(676*128, 4096),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(4096, 1024),
            ReLU(inplace=True),
            Linear(1024, 256),
            ReLU(inplace=True),
            Linear(256, 8)
        )

    def forward(self, x):
        out = self.layer1(x)
        # print('first layer output {}:{}'.format(out.shape, out))
        out = self.layer2(out)
        # print('second layer output {}:{}'.format(out.shape, out))
        out = self.layer3(out)
        # print('third layer output {}:{}'.format(out.shape, out))
        out = out.view(out.size(0), -1)
        # print('third1 layer output {}'.format(out.shape))
        out = self.fc(out)
        return out

# train the model
def train_model(train_dl, model, device, epochs):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=Leaning_Rate, momentum=0.9)                  #?????????????????????????????  how t set the parameters
    # enumerate epochs
    for epoch in range(epochs):
        # myloss = zeros(len(train_dl.dataset))
        myloss = 0.
        counter = 0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):

            counter += 1
            inputs, targets = inputs.to(device), targets.to(device)

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)

            # calculate loss
            #y_targets = tensor(targets, dtype=torch.long, device=device)
            targets = targets.long()
            loss = criterion(yhat, targets)
            myloss += loss.detach().cpu().numpy()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        print('The mean loss of epoch ' + str(epoch) + ' is: %.3f' % (myloss/counter))

# evaluate the model
def evaluate_model(test_dl, model, device):
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
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
    return acc

def active_learning(csvfile, model, device):

    dataCSV = Data_Path + csvfile

    # al_train_data = ImageDataset(dataCSV)
    al_train_data = CSVDataset(dataCSV)
    al_dl = DataLoader(al_train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    data = read_csv(dataCSV, sep=',', header=None)

    new_labels =list()
    for i, (inputs, targets) in enumerate(al_dl):
        # data_labels = DataFrame(data).values[i:i+BATCH_SIZE, -3]
        # data_labels = data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE, -3]

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
        # expert annotation
        actual = targets.numpy()
        # sample selection function: If the confidence score is blew the threshold, ask the annotator to manually annotate that sample (in practice, just take the annotation from this sample directly from the dataset)
        for j in range(len(yhat)):
            if yhat_prob[j] < 0.125:
                yhat_cls[j] = actual[j]
            new_labels.append(yhat_cls[j])
    data.iloc[:, -3] = new_labels

    al_dataset_csv = Data_Path + 'AL_dataset.csv'
    if os.path.exists(al_dataset_csv):
        os.remove(al_dataset_csv)
    data.to_csv(al_dataset_csv, index=False, header = False, encoding='utf-8')
    # with open(al_dataset_csv,'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in update_AL_set:
    #         writer.writerow(row)

    # AL_dataset = ImageDataset(Data_Path+'AL_dataset.csv')
    AL_dataset = CSVDataset(Data_Path+'AL_dataset.csv')
    AL_dl =  DataLoader(AL_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    return  AL_dl

def main():

    # time
    print("Started time:{}".format(startTime))
    # prepare the data
    # train_data_path = 'C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/FAUs/training_with_discard.csv'
    # test_data_path = 'C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/FAUs/validation_with_discard.csv'
    # train_data_path = 'C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/RAW/training_with_discard.csv'
    # test_data_path = 'C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/RAW/validation_with_discard.csv'
    # train_data_path = '/nas/student/QiangChang/AffectNet/FAUs/training_with_discard.csv'
    # test_data_path = '/nas/student/QiangChang/AffectNet/FAUs/validation_with_discard.csv'
    # train_data_path = '/nas/student/QiangChang/AffectNet/RAW/training_with_discard.csv'
    # test_data_path = '/nas/student/QiangChang/AffectNet/RAW/validation_with_discard.csv'
    train_data_path = 'training_with_discard.csv'
    train_data_pathA = 'training_with_discard_A.csv'
    train_data_pathB = 'training_with_discard_B.csv'
    val_data_path = 'validation_with_discard.csv'
    # train_data_path = '/nas/student/QiangChang/AffectNet/FAUs/training.csv'
    # test_data_path = '/nas/student/QiangChang/AffectNet/FAUs/validation.csv'
    # train_data_path = '/nas/student/QiangChang/AffectNet/RAW/training.csv'
    # test_data_path = '/nas/student/QiangChang/AffectNet/RAW/validation.csv'
    # demo_path = 'C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/FAUs/demo1.csv'

    # train_data = CSVDataset(Data_Path+train_data_path)
    # val_data = CSVDataset(Data_Path+val_data_path)
    train_data = ImageDataset(Data_Path+train_data_path)
    val_data = ImageDataset(Data_Path+val_data_path)

    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)                           #??????????????????????????????????????????batch_size
    val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    print('-------  Data Information  -------')
    print("Size of train and test data: {},{}" .format(len(train_dl.dataset), len(val_dl.dataset)))

    # for CSVDataset
    # train = list(set(train_data.y))
    # val = list(set(val_data.y))
    #for ImageDataset
    train = list(set(train_data.labels))
    val = list(set(val_data.labels))
    print('-------  Label Information  -------')
    print("The class labels of train data: " + str(train))
    print("The class labels of  test data: " + str(val))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the network
    # model = MLP1(17).to(device)
    # model = MLP(136).to(device)
    # model = CNND().to(device)
    model = CNN2D().to(device)
    # train the model
    train_model(train_dl, model, device, 40)
    # evaluate the model
    acc = evaluate_model(val_dl, model, device)

    print("Evaluating the Model:{} with DataSet={}, Epochs={}, lr={} and Batch_size={}".format(model._get_name(), DATA_SET, EPOCHS, Leaning_Rate, BATCH_SIZE))
    print('Accuracy after first training: %.3f' % acc)

    if acc > 0.429 :
        torch.save(model, Data_Path + 'Visaul_Model.pt')
    # # train the optimal model using data belonging to the Subset A of the training data and the development data
    # list_of_datasets = []
    # list_of_datasets.append(train_data)
    # list_of_datasets.append(val_data)
    # TD_dataset = ConcatDataset(list_of_datasets)
    # # print(TD_dataset.__len__())
    #
    # train_dl = DataLoader(TD_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    #
    # print("----------------------Second training------------------------")
    # train_model(train_dl, model, device)
    #
    # acc = evaluate_model(val_dl, model, device)
    # print('Accuracy after second training: %.3f' % acc)

    # AL_dl = active_learning(train_data_pathB, model, device)
    # # re-train the model
    # print("----------------------Third training(AL)------------------------")
    # train_model(AL_dl, model, device, 200)
    # # assess the model performance
    # acc = evaluate_model(val_dl, model, device)
    # print('Accuracy after third training: %.3f' % acc)

    # make a single prediction
    # row = [5.1,3.5,1.4,0.2]
    # yhat = predict(row, model)
    # print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))


    endTime = datetime.datetime.now()
    duration  = endTime - startTime
    seconds = duration.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print("The program has run " + '{} hours, {} minutes, {} seconds'.format(hours, minutes, seconds))



if __name__ == "__main__":
    main()




# Evaluating
#      Task                     Model     Epochs     Lr        Batch_size       acc1                     Time            threshold      batch2       acc2
#  Emotion(Raw Image-10000)     CNN2d      40       0.001        48            39.8%                  17h 42min
#  Emotion(Raw Image-All)       CNN2d      40       0.001        48            43.5%                  42h 15min --------------------------------------------------------------
#  Emotion(Raw Image-All)       CNN2d      40       0.01         48            41.0%                  42h 47min
#  Emotion(Raw Image-All)       CNN2d      50       0.01         32            42.8%                  67h 3min

#  Emotion(Raw Image-All)       CNN2d      40       0.001        48            43.3%                  32h 36min           0.125          5          41.5%
#  Emotion(Raw Image-All)       CNN2d      40       0.001        48            43.1%                  33h 32min           0.5            5          41.6%

#  Emotion(Landmarks)           CNN2d      250      0.01         32            24.4%                  2h 8min
#  Emotion(Landmarks)           CNN2d      250      0.005        32            24.5%                  2h 9min --------------------------------------------

#  Emotion(FAUs)                 MLP       200      0.01         16            22.1%                  32min
#  Emotion(FAUs)                 MLP       100      0.01         32            24.2%                  24min
#  Emotion(FAUs)                 MLP1      100      0.01         32            24.6%                  36min ------------------------------------------------
#  Emotion(FAUs)                cnn1d      100      0.01         32            25.6%                  55min ----------------------------------------------
#  Emotion(FAUs)                cnn2d      100      0.01         32            25.2%                  51min -----------------------------------------------
