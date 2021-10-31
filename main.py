import cv2
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = ["A","B","C","D","E","F","G","H","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nClasses = len(classes)

xTrain, xTest, yTrain, yTest = tts(X, y, test_size=0.25, random_state=36)
ss = StandardScaler()
xTrain = ss.fit_transform(xTrain)
xTest = ss.fit_transform(xTest)
model = LogisticRegression(random_state=0)
model.fit(xTrain, yTrain)
yPredict = model.predict(xTest)
accuracy = acc(yTest, yPredict)
print(accuracy)

cap = cv2.VideoCapture(0)

while (True) :
  try :
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    im_pil = Image.fromarray(roi)

    imageBW = im_pil.convert('L')
    imageResized = imageBW.resize((28,28), Image.ANTIALIAS)

    imageInverted = PIL.ImageOps.invert(imageResized)
    pixel_filter = 20
    min_pixel = np.percentile(imageInverted, pixel_filter)
    imageScaled = np.clip(imageInverted-min_pixel, 0, 255)
    max_pixel = np.max(imageInverted)
    imageScaled = np.asarray(imageScaled)/max_pixel
    test_sample = np.array(imageScaled).reshape(1,784)
    test_pred = model.predict(test_sample)
    print("Predicted class is: ", test_pred)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

cap.release()
cv2.destroyAllWindows()