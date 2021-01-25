__author__ = 'ASUS'

#preparing data path
basePath_Data = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/ComParE2013_Emotion/dist/"
basePath_openSMILE = "E:/cq/opensmile-2.3.0/"

dataPath = basePath_Data + "wav/"
openSMILE_Path = basePath_openSMILE + "bin/Win32/"
# configFile_Path = basePath_openSMILE + "config/gemaps/eGeMAPSv01a.conf"
configFile_Path = basePath_openSMILE + "config/IS13_ComParE.conf"

import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreProcessing:
    """extract features from wav_file using opensmile"""
    def __init__(self):
        self.data = []

    def extract_features(self):
        files = os.listdir(dataPath)
        #print(files)
        #os.system(basePath_openSMILE + "bin/Win32/SMILExtract_Release -h")
        # train = 0
        # test= 0
        # devel =0
        for file in files:
            if 'train' in file.title().lower():
                cmdstr = openSMILE_Path + "SMILExtract_Release -C " + configFile_Path +" -instname " +file.title().lower()+ " -appendcsv 1 -I \"" + dataPath + file + "\" -csvoutput \"" + basePath_Data + "ComParE_training.csv\""
            elif 'test' in file.title().lower():
                cmdstr = openSMILE_Path + "SMILExtract_Release -C " + configFile_Path +" -instname " +file.title().lower()+ " -appendcsv 1 -I \"" + dataPath + file + "\" -csvoutput \"" + basePath_Data + "ComParE_test.csv\""
                #test = test + 1
            else:
                cmdstr = openSMILE_Path + "SMILExtract_Release -C " + configFile_Path +" -instname " +file.title().lower()+ " -appendcsv 1 -I \"" + dataPath + file + "\" -csvoutput \"" + basePath_Data + "ComParE_devel.csv\""
            os.system(cmdstr)
        #print(train,test,devel)

    def extract_feature(self, index_speaker):
        files = os.listdir(dataPath)
        for file in files:
            if index_speaker in os.path.splitext(file)[0][0:2]:
                cmdstr = openSMILE_Path + "SMILExtract_Release -C " + configFile_Path +" -instname " +os.path.splitext(file)[0]+ " -appendcsv 1 -I \"" + dataPath + file + "\" -csvoutput \"" + basePath_Data + "EmoDB_test.csv\""
            else:
                cmdstr = openSMILE_Path + "SMILExtract_Release -C " + configFile_Path +" -instname " +os.path.splitext(file)[0]+ " -appendcsv 1 -I \"" + dataPath + file + "\" -csvoutput \"" + basePath_Data + "EmoDB_train.csv\""
            os.system(cmdstr)

    def read(self):
        db = pd.read_csv(basePath_Data + "lab/ComParE2013_Emotion.tsv",sep="\t")
        print(pd.DataFrame(db).values[215,3])
        print(pd.DataFrame(db).values[217,3])

    def reshapeCSV(self, old, new):

        df = pd.read_csv(old, sep=';')
        names = pd.DataFrame(df).values[:,0]            #wavfile
        rows = pd.DataFrame(df).values[:,:].tolist()    #frames
        lists = []
        for name in sorted(list(set(names))):
            ls = []
            ls.append(name)
            for row in rows:
                if row[0] == name:
                    ls.extend(row[2:])
                    # for i in row[2:]:
                    #     ls.append(i)

            frame_num = (len(ls)-1)/23   ##the first value is the name of file, doesn't belong to features; each frame has 23 feature values
            if frame_num < 248: #padding when the number of frames smaller than the mean number of frames,
                for i in range(248 - int(frame_num)):  #frames which need to be padding
                    ls.extend([0]*23)                #each frame has 23 feature values, therefore padding with 23 zeros
                    # for i in range(23):            #each frame has 23 feature values, therefore padding with 23 zeros
                    #     ls.append(0)#ls.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            if frame_num > 248: #discard when the number of frames bigger than the mean number of frames, keep the middle 248 frames, which are more representational for the voice
                # ls = ls[:248*23+1]
                temp = []
                temp.append(ls[0])
                mid = frame_num/2
                left = int(mid - 124)*23+1
                right = int(mid + 124)*23+1
                temp.extend(ls[left:right])
                ls = temp
            lists.append(ls)
        # print(len(lists[199]))
        with open(new, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in lists:
                # for element in row:
                writer.writerow(row)
        print("rewrite csv finished")


    def splitCSV(self, data_path):

        # #read audio data
        # df = pd.read_csv(data_path, sep=';')
        # data = pd.DataFrame(df).values
        # df_columns = df.columns.values.tolist()  #herder

        #read image data
        df = pd.read_csv(data_path, header = None)
        data = pd.DataFrame(df).values

        subA, subB = train_test_split(data, test_size=0.25, random_state=0)
        # print(subA.shape)
        # print(subA[0])
        # print(subB.shape)
        # print(subB[-1])
        subA = subA.tolist()
        subB = subB.tolist()

        (file, ext) = os.path.splitext(data_path)
        newA = file + "_A" + ext
        newB = file + "_B" + ext


        with open(newA,'w', newline='') as csvfileA:
            writer = csv.writer(csvfileA)
            for row in subA:
                writer.writerow(row)

        with open(newB,'w', newline='') as csvfileB:
            writer = csv.writer(csvfileB)
            for row in subB:
                writer.writerow(row)

        # csv.register_dialect('mydialect', delimiter=';')
        # with open(newA,'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile, 'mydialect')
        #     writer.writerow(df_columns) #for audio data set
        #     for row in subA:
        #         writer.writerow(row)
        #
        # with open(newB,'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile, 'mydialect')
        #     writer.writerow(df_columns) #for audio data set
        #     for row in subB:
        #         writer.writerow(row)
        #
        # csv.unregister_dialect('mydialect')
        print("spliting has done!")


pp = PreProcessing()
# pp.extract_features()
# pp.read()

# train_old = basePath_Data + 'EmoDB_training.csv'
# train_new = basePath_Data + 'EmoDB_lld_training.csv'
# devel_old = basePath_Data + 'EmoDB_devel_.csv'
# devel_new = basePath_Data + 'EmoDB_lld_devel.csv'
# test_old = basePath_Data + 'EmoDB_test.csv'
# test_new = basePath_Data + 'EmoDB_lld_test.csv'
#
# pp.reshapeCSV(train_old, train_new)
# pp.reshapeCSV(devel_old, devel_new)
# pp.reshapeCSV(test_old, test_new)

#active learning(divide training data set into two fixed parts)
##audio data set
# training_data = basePath_Data + 'ComParE_training.csv'
##image data set
training_data = "C:/Users/ASUS/Desktop/Uni-Bewerbung/Augsburg/5.Semester/MasterArbeit/DB/AffectNet/FAUs/training_with_discard.csv"
pp.splitCSV(training_data)