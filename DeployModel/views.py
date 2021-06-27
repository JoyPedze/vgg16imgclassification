from django.http import HttpResponse
from django.shortcuts import render
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
# %matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images


# Create your views here.
def home(request):
    return render(request, "home.html")

def result(request):
        count = 0
        uservideo = (request.GET['Video'])
        videoFile = uservideo
        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                filename ="frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()

        from keras.applications.vgg16 import VGG16
        # load the model
        model = VGG16()

        from keras.preprocessing.image import load_img
        # load an image from file
        image = load_img('frame0.jpg', target_size=(224, 224))
        from keras.preprocessing.image import img_to_array
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        from keras.applications.vgg16 import preprocess_input
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        yhat = model.predict(image)
        from keras.applications.vgg16 import decode_predictions
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        # print the classification
        result = ((label[1]))
        a = []
        a.append(result)

        from keras.preprocessing.image import load_img
        # load an image from file
        image1 = load_img('frame1.jpg', target_size=(224, 224))
        from keras.preprocessing.image import img_to_array
        # convert the image pixels to a numpy array
        image1 = img_to_array(image1)
        # reshape data for the model
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        from keras.applications.vgg16 import preprocess_input
        # prepare the image for the VGG model
        image1 = preprocess_input(image1)
        # predict the probability across all output classes
        yhat1 = model.predict(image1)
        from keras.applications.vgg16 import decode_predictions
        # convert the probabilities to class labels
        label1 = decode_predictions(yhat1)
        # retrieve the most likely result, e.g. highest probability
        label1 = label1[0][0]
        # print the classification
        result1 = ((label1[1]))
        a.append(result1)

        from keras.preprocessing.image import load_img
        # load an image from file
        image2 = load_img('frame2.jpg', target_size=(224, 224))
        from keras.preprocessing.image import img_to_array
        # convert the image pixels to a numpy array
        image2 = img_to_array(image2)
        # reshape data for the model
        image2 = image2.reshape((1, image2.shape[0], image2.shape[1], image2.shape[2]))
        from keras.applications.vgg16 import preprocess_input
        # prepare the image for the VGG model
        image2 = preprocess_input(image2)
        # predict the probability across all output classes
        yhat2 = model.predict(image2)
        from keras.applications.vgg16 import decode_predictions
        # convert the probabilities to class labels
        label2 = decode_predictions(yhat2)
        # retrieve the most likely result, e.g. highest probability
        label2 = label2[0][0]
        # print the classification
        result2 = ((label2[1]))
        a.append(result2)

        from keras.preprocessing.image import load_img
        # load an image from file
        image3 = load_img('frame3.jpg', target_size=(224, 224))
        from keras.preprocessing.image import img_to_array
        # convert the image pixels to a numpy array
        image3 = img_to_array(image3)
        # reshape data for the model
        image3 = image3.reshape((1, image3.shape[0], image3.shape[1], image3.shape[2]))
        from keras.applications.vgg16 import preprocess_input
        # prepare the image for the VGG model
        image3 = preprocess_input(image3)
        # predict the probability across all output classes
        yhat3 = model.predict(image3)
        from keras.applications.vgg16 import decode_predictions
        # convert the probabilities to class labels
        label3 = decode_predictions(yhat3)
        # retrieve the most likely result, e.g. highest probability
        label3 = label3[0][0]
        # print the classification
        result3 = ((label3[1]))
        a.append(result3)

        def search(list, platform):
            for i in range(len(list)):
                if list[i] == platform:
                    return True
            return False

        name_of_object = (request.GET['Object'])

        if search(a, name_of_object):
            ans = "is found"
        else:
            ans = "is not found"




    # LA = joblib.load('finalized_model.sav')

    # s1 = pd.Series([(request.GET['Tenure']),(request.GET['Dependents']),(request.GET['MultipleLines']),(request.GET['InternetService']),(request.GET['PhoneService']),(request.GET['PaymentMethod']),(request.GET['TotalCharges']),(request.GET['Contract']),(request.GET['StreamingTV']),(request.GET['OnlineBackup'])])
    # cols =  ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # df = pd.DataFrame([list(s1)],  columns =  cols)

    # lis = []

    # lis.append(request.GET['Tenure'])
    # lis.append(request.GET['Dependents'])
    # lis.append(request.GET['MultipleLines'])
    # lis.append(request.GET['InternetService'])
    # lis.append(request.GET['PhoneService'])
    # lis.append(request.GET['PaymentMethod'])
    # lis.append(request.GET['TotalCharges'])
    # lis.append(request.GET['Contract'])
    # lis.append(request.GET['StreamingTV'])
    # lis.append(request.GET['OnlineBackup'])

    # ans = LA.predict(df)

        return render(request,"result.html",{'ans':ans})