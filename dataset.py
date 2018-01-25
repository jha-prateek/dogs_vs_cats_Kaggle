import cv2
from tqdm import tqdm
import os
from random import shuffle
import numpy as np

train_dir = "C:/Users/prate/Downloads/Compressed/dogs_vs_cats/train"
test_dir = "C:/Users/prate/Downloads/Compressed/dogs_vs_cats/test"
img_size = 50

def label_img(img):
    label = img.split(".")[-3]
    if(label == "cat"):
        return [1,0]
    elif(label == "dog"):
        return [0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save("training.npy", training_data)
    return training_data

def preocess_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img.split(".")[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save("testing.npy", testing_data)
    return testing_data

def load_test_data():
    if(os.path.exists("testing.npy")):
        return np.load("testing.npy")
    else:
        return preocess_test_data()

def load_train_data():
    if (os.path.exists("training.npy")):
        return np.load("training.npy")
    else:
        return create_train_data()