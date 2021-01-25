__author__ = 'ASUS'

import os
import io
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import torch
from torchvision import transforms

Emo_labels = {0:'Neutral',1:'Happiness',2:'Sadness',3:'Surprise',4:'Fear',5:'Disgust',6:'Anger',7:'Contempt'}
develFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/RAW/validation.csv"
# develFile1 = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/FAUs/demo1.csv"
# train_data_path = '/nas/student/QiangChang/AffectNet/FAUs/training.csv'
# test_data_path = '/nas/student/QiangChang/AffectNet/FAUs/validation.csv'
#
# train_path = '/nas/student/QiangChang/AffectNet/FAUs/training1.csv'
# test_path = '/nas/student/QiangChang/AffectNet/FAUs/validation1.csv'

trainFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/ComParE_devel.csv"

import csv

def discard_irrel_cls(data_path, new):
    df = read_csv(data_path, sep=';', header=None)
    labels = DataFrame(df).values[:,-3]
    for i, label in enumerate(labels):
        if label not in Emo_labels.keys():
            df = df.drop([i])
    ls = df.values.tolist()

    # print(list)
    with open(new,'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in ls:
            writer.writerow(row)

    print("transformation finished")
# discard_irrel_cls(train_data_path, train_path)
#
# discard_irrel_cls(test_data_path, test_path)

import cv2
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt



img = 'C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/RAW/cfa0c679da3dbe9f01e92cdeda2da1065aa50e7bf0b3990ec335f726.jpg'
# with open(img, 'rb') as f:    # ValueError: seek of closed file   #Too many open files
#     img_ = np.asarray(Image.open(io.BytesIO(f.read())))
# img_ = np.asarray(Image.open(img))
img_ = Image.open(img)
image_array = np.asarray(Image.open(img))

# print(img_[1:15,1:15])
# print(type(img_))
# print(img_.shape)
# image_array = cv2.imread(img)
# img_arr = image_array.flatten()
# img_arr = img_arr.astype('float32')
# img_arr /= 255.0
# img_arr = img_arr.reshape(image_array.shape[0],image_array.shape[1],image_array.shape[2])
im = Image.fromarray(image_array)
# print(img_arr)

df = read_csv(develFile, sep=',')
# # print(len(df))
fileName = DataFrame(df).values[:,0][1]
loc_x = DataFrame(df).values[:,1][1]
loc_y = DataFrame(df).values[:,2][1]
width = DataFrame(df).values[:,3][1]
height = DataFrame(df).values[:,4][1]
print(fileName)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(2, 2, 1)
# ax.title.set_text('Original image')
# ax.imshow(img_)    #----------------------the original image
# a = np.asarray(img_)
# # print("original:{}".format(a))
# # print(a.shape)
# b = a[loc_x:loc_x+width, loc_y:loc_y+height, ]
# img1 = transforms.ToPILImage()(b)
# # print("bounding:{}".format(np.asarray(img1)))
# # print(np.asarray(img1).shape)
# ax = fig.add_subplot(2, 2, 2)  #----------face in the bounding box
# ax.title.set_text('Face in the bounding box')
# ax.imshow(img1)
# img2 = transforms.Resize(224)(img1)
# # print("resize:{}".format(np.asarray(img2)))
# # print(np.asarray(img2).shape)
# ax = fig.add_subplot(2, 2, 3)
# ax.title.set_text('Face resized to 224')
# ax.imshow(img2)         #-----------------face resized to 224
#
# # img3 = np.asarray(img2)
# img3 = transforms.ToTensor()(img2) #torch.from_numpy(np.asarray(img2)) #img3 = img3.type('torch.FloatTensor')
# # print(torch.max(img3,2)[0])
# # print(torch.min(img3,2)[0])
# # print(torch.get_default_dtype(img3))
# # img3 = img3.permute(2,0,1)
# img3 = transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])(img3)
# # img3 = torch.clamp(img3,0,1)
# # print(torch.max(img3,2)[0])
# # print(torch.min(img3,2)[0])
# # print("Bofore Normal:{}".format(img3))
# # max1 = torch.max(img3,2)[0]
# # max = torch.max(max1,1)[0]
# # min1 = torch.min(img3,2)[0]
# # min = torch.min(min1,1)[0]
# # neg_min = torch.mul(min, -1)
# # print(max)
# # print(neg_min)
# # for i in range(img3.size()[0]):
# #     img3[i] = torch.add(img3[i],neg_min[i].item())
# #     # print(img3[i])
# #     max_min = torch.add(max[i],neg_min[i].item()).item()
# #     # print(max_min)
# #     img3[i] = torch.div(img3[i], max_min)
# # print("After Normal:{}".format(img3))
# # print(img3.size())
# # img3 = torch.div(torch.add(img3,neg_min),torch.add())
# img3 = transforms.ToPILImage()(img3)
# ax = fig.add_subplot(2, 2, 4)
# ax.title.set_text('Normalisation')
# ax.imshow(img3)       #------------------face after normalized
# plt.show()

pixels = DataFrame(df).values[:,5]
# print(type(pixels))
# print(len(pixels))
arr = [float(x) for x in pixels[1].split(";")]
landmarks = np.array(arr).reshape(68,2)#.swapaxes(1,0)
# print(landmarks)
# for landmark in landmarks:
#     landmark[0] = (landmark[0]-loc_x)/width
#     landmark[1] = (landmark[1]-loc_y)/height
# print(landmarks)

# img_arr = image_array.flatten()
# img_arr = img_arr.astype('float32')
# img_arr = (img_arr - loc_x)/width
# img_arr = img_arr.reshape(image_array.shape[0],image_array.shape[1],image_array.shape[2])

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1)
ax.title.set_text('Original image')
ax.imshow(img_)
ax = fig.add_subplot(1, 3, 2)
ax.title.set_text('68 landmarks')
ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
ax = fig.add_subplot(1, 3, 3)
ax.title.set_text('Landmarks on face')
img2 = image_array.copy()
for p in landmarks:
    img2[int(p[1])-3:int(p[1])+3, int(p[0])-3:int(p[0])+3, :] = (255, 255, 255)
ax.imshow(img2)
# plt.show()

# plt.imshow(img_)
plt.show()
# visualize_landmark(image_array, arr)
#
# labels = DataFrame(df).values[:,0]
# print(len(list(set(labels))))
# result = pd.value_counts(labels)
# a = sorted(result)
# print(result)
# print(type(result))
# for i, label in enumerate(labels):
#     if label not in Emo_labels.keys():
#         df = df.drop([i])
# ls = df.values.tolist()
# print(len(ls))



def reshapeCSV(old, new):

    df = read_csv(old, sep=';')
    labels = DataFrame(df).values[:,0]
    rows = DataFrame(df).values[:,:]
    lists = []
    for name in sorted(list(set(labels))):
        ls = []
        for row in rows:
            if row[0] == name:
                ls.append(row[2:])
        arr = np.array(ls)
        if arr.shape[0] < 248:
            for i in range(248 - arr.shape[0]):
                ls.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if arr.shape[0] > 248:
            # for i in range(arr.shape[0] - 248):
            ls = ls[:248]
        lists.append(ls)

    with open(new, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in lists:
            writer.writerow(row)








def getData(file):
    df = pd.read_csv(file, sep=',')
    # discard_irrel_cls(df)

    loc_x = pd.DataFrame(df).values[:,1]
    loc_y = pd.DataFrame(df).values[:,2]
    width = pd.DataFrame(df).values[:,3]
    height = pd.DataFrame(df).values[:,4]
    pixels = pd.DataFrame(df).values[:,5]
    labels = pd.DataFrame(df).values[:,-3]

    print(pixels[0])
    features = []
    for i in range(len(pixels)):
        arr = [float(x) for x in pixels[i].split(";")]
        landmarks = np.array(arr).reshape(68,2)#.swapaxes(1,0)
        # print(landmarks)
        for landmark in landmarks:
            landmark[0] = (landmark[0]-loc_x[i])/width[i]
            landmark[1] = (landmark[1]-loc_y[i])/height[i]
        features.append(landmarks.flatten())
    features = np.array(features)
    # print(features.shape)
    return features, labels

# train,label = getData(develFile)
# print(train[0])
# print(set(label))

from torch.utils.data import Dataset

# transform1 = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])])

class ImageDataset(Dataset):

    def __init__(self, csvfile):

        # path = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/RAW/"
        path = '/nas/student/QiangChang/AffectNet/RAW/'
        image_path = path + "Manually_Annotated_Images/"
        dataCSV = path + csvfile

        # df = pd.read_csv(dataCSV, sep=',')
        df = pd.read_csv(dataCSV, sep=',', header=None) #irrelevant discard files
        file_names = DataFrame(df).values[:, 0]
        img_bounding_box = DataFrame(df).values[:, 1:5]
        file_labels = DataFrame(df).values[:, -3]

        loadedImages = []
        for name in file_names:
            # with open(os.path.join(image_path, name), 'rb') as f:    # ValueError: seek of closed file   #Too many open files
            #     img = Image.open(io.BytesIO(f.read()))      #OSError: [Errno 14] Bad address       slurmstepd: Exceeded step memory limit at some point.
            loadedImages.append(os.path.join(image_path, name))
        self.images = loadedImages
        self.bounding_box = img_bounding_box
        self.labels = file_labels
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])])

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        imgArray = np.asarray(img)
        face_loc_x = self.bounding_box[idx][0]
        face_loc_y = self.bounding_box[idx][1]
        face_width = self.bounding_box[idx][2]
        face_height = self.bounding_box[idx][3]
        faceArray = imgArray[face_loc_x:face_loc_x+face_width+1, face_loc_y:face_loc_y+face_height+1,]
        return [self.transforms(faceArray), self.labels[idx]]

    def __len__(self):
        return len(self.images)

# train_dataset = ImageDataset("training_with_discard.csv")
# print(train_dataset.__len__())
# print(train_dataset.__getitem__(0))
# print(train_dataset[0][0].size())
# # print(train_dataset[0][1])
#
# val_dataset = ImageDataset("validation_with_discard.csv")
# print(val_dataset.__len__())
# print(val_dataset.__getitem__(0))
# print(val_dataset[0][0].size())


import arff
valCSV = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/RAW/validation_with_discard.csv"
labelFile = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/lab/ComParE2013_Emotion.test.lab.csv"
labelFile1 = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/lab/ComParE2013_Emotion.test.lab1.csv"
# data = read_csv(valCSV, sep=',', header=None)

def del_csv_col(fname,newfname,idxs):
    with open(fname) as csvin, open(newfname, 'w', newline='') as csvout:
        reader = csv.reader(csvin)
        writer = csv.writer(csvout)
        rows = (tuple(item for idx, item in enumerate(row) if idx not in idxs) for row in reader)
        writer.writerow(rows)

# del_csv_col(labelFile, labelFile1, [0])

# data= pd.read_csv(labelFile)
# print(data.values[:,0])