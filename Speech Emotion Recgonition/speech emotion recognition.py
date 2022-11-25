import matplotlib.pyplot as plt                      
import librosa  as lib                                     
import librosa.display                              
import os                                             
import scipy.io.wavfile                               
import numpy as np                                    
import fastai
import glob                                                                                                  


from fastai import *                                 
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.tabular.all import *
from fastai.text.all import *
from fastai.vision.widgets import *


import struct                                         
import time
from tkinter import TclError
from scipy.fftpack import fft                         
import sounddevice
from scipy.io.wavfile import write

class FetchLabel():

    def get_emotion(self, file_path):
        item = file_path.split('/')[-1]
        if item[6:-16]=='02' and int(item[18:-4])%2==0:
            return 'female_calm'
        elif item[6:-16]=='02' and int(item[18:-4])%2==1:
            return 'male_calm'
        elif item[6:-16]=='03' and int(item[18:-4])%2==0:
            return 'female_happy'
        elif item[6:-16]=='03' and int(item[18:-4])%2==1:
            return 'male_happy'
        elif item[6:-16]=='04' and int(item[18:-4])%2==0:
            return 'female_sad'
        elif item[6:-16]=='04' and int(item[18:-4])%2==1:
            return 'male_sad'
        elif item[6:-16]=='05' and int(item[18:-4])%2==0:
            return 'female_angry'
        elif item[6:-16]=='05' and int(item[18:-4])%2==1:
            return 'male_angry'
        elif item[6:-16]=='06' and int(item[18:-4])%2==0:
            return 'female_fearful'
        elif item[6:-16]=='06' and int(item[18:-4])%2==1:
            return 'male_fearful'
        elif item[6:-16]=='01' and int(item[18:-4])%2==0:
            return 'female_neutral'
        elif item[6:-16]=='01' and int(item[18:-4])%2==1:
            return 'male_neutral'
        elif item[6:-16]=='07' and int(item[18:-4])%2==0:
            return 'female_disgusted'
        elif item[6:-16]=='07' and int(item[18:-4])%2==1:
            return 'male_disgusted'
        elif item[6:-16]=='08' and int(item[18:-4])%2==0:
            return 'female_surprised'
        elif item[6:-16]=='08' and int(item[18:-4])%2==1:
            return 'male_surprised'
        elif item[:1]=='a':
            return 'male_angry'
        elif item[:1]=='f':
            return 'male_fearful'
        elif item[:1]=='h':
            return 'male_happy'
        elif item[:1]=='n':
            return 'male_neutral'
        elif item[:2]=='sa':
            return 'male_sad'
        elif item[:1]=='d':
            return 'male_disgusted'
        elif item[:2]=='su':
            return 'male_surprised'

label = FetchLabel() 

AUDIO_FOLDER = "D:/mini project/audio-dataset/*"
OUTPUT_FOLDER_TRAIN = "D:/mini project/output_folder_train/"
OUTPUT_FOLDER_TEST = "D:/mini project/output_folder_test/"

data, sampling_rate = librosa.load('D:/mini project/audio-dataset/Actor_01/03-01-01-01-01-01-01.wav')
plt.figure(figsize=(40, 5))                           
librosa.display.waveshow(data, sr=sampling_rate)   

y, sr = librosa.load('D:/mini project/audio-dataset/Actor_01/03-01-01-01-01-01-01.wav')
yt,_=librosa.effects.trim(y)

audio_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)

audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)

librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')

for actor in glob.glob(AUDIO_FOLDER):              
  for name in glob.glob(actor +'/*'):              
    print(name[-18:-16])                           
    emotion = label.get_emotion(name[-24:])        
    print(emotion) 

dicts={'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}

counts = {}
for actor in glob.glob(AUDIO_FOLDER):               
  for audioFile in glob.glob(actor +'/*'):          
    emotion=dicts[audioFile[-18:-16]]               
    save_path_train = OUTPUT_FOLDER_TRAIN + emotion 
    save_path_test = OUTPUT_FOLDER_TEST + emotion   
    
    y, sr = librosa.load(audioFile)                 
    yt,_=librosa.effects.trim(y)                      
    y=yt
    
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=20000, x_axis='time')
    
    
    count = counts.get(emotion, 1)
    if (count % 9 == 0):
      p = os.path.join(save_path_test, "{}{}.jpg".format(emotion, str(count).zfill(6)))
    else:
      p = os.path.join(save_path_train, "{}{}.jpg".format(emotion, str(count).zfill(6)))              
    counts[emotion] = count + 1

    
    plt.savefig(p)
    print("Done!")
print("Done actor!")

img = plt.imread('D:/mini project/output_folder_train/sorted_data/angry/angry000001.jpg')   
plt.imshow(img)

train_path = Path("D:/mini project/output_folder_train/sorted_data")
valid_path = Path("D:/mini project/output_folder_test")
main_path = Path('D:/mini project/audio-dataset')

train_path.ls()

dls = ImageDataLoaders.from_folder(train_path, valid_pct=0.2, seed=42, num_workers=0)
dls.valid_ds.items[:10]

dls.vocab
