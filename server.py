import os
import torch
from numpy import array, asarray
from pandas import read_csv, DataFrame
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torchvision import transforms


openSMILE_Path = "#to do#"
configFile_Path = "#to do#"

audio_model_path = "#to do#"
visual_model_path = "#to do#"

class Server:
    def extract_features(self, file):
        """
            used for extracting feautres from a single audio file recorded in the demo
        """
        cmdstr = openSMILE_Path + "SMILExtract_Release -C " + configFile_Path + " -instname " + file.title().lower() + " -appendcsv 1 -I \"" + file + "\" -csvoutput \"" + "demo.csv\""
        os.system(cmdstr)

    def predict(self, input):
        """input: the path to the .csv file or the .png/.jpg file"""

        result = ''

        file_name, file_type = os.path.splitext(input)

        if file_type == '.csv': #feature of the audio file
            df_f = read_csv(input, sep=';')
            features = df_f.values[:, 2:]
            # Standardize the features using sklearn
            scaler = MinMaxScaler().fit(features)
            features = scaler.transform(features)
            # ensure input data is floats
            features = features.astype('float32')
            # ensure type of input data is tensor
            feature_X = torch.tensor(features)

            audio_model = torch.load(audio_model_path)
            prediction = audio_model(feature_X)
            result = self.speech_label(prediction)

        else: #image file
            img = Image.open(input)
            imgArray = asarray(img)

            transformOp = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
            ])
            image_X = transformOp(imgArray)

            visual_model = torch.load(visual_model_path)
            prediction = visual_model(image_X)
            result = self.face_label(prediction)

        return result

    def speech_label(self, prediction):
        # the dict here is just a sketch, we don't know the encoder of these labels, should be re-edit
        Emo_labels_dict = {0: 'elation', 1: 'pleasure', 2: 'relief', 3: 'anxiety', 4: 'amusement', 5: 'hot anger', 6: 'cold anger',
                      7: 'despair', 8: 'interest', 9: 'pride', 10: 'sadness', 11: 'panic fear'}
        return Emo_labels_dict.get(prediction, None)

    def face_label(self, prediction):

        Emo_labels_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger',
                      7: 'Contempt'}
        return Emo_labels_dict.get(prediction, None)
