import sklearn
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import cv2
import pickle
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split

from google.colab.patches import cv2_imshow

labels = {'astilbe': 1,'bellflower': 2,'black_eyed_susan': 3,'calendula': 4,'california_poppy': 5,'carnation': 6,'common_daisy': 7,'coreopsis': 8,
          'daffodil': 9,'dandelion': 10,'iris': 11,'magnolia': 12,'rose': 13,'sunflower': 14,'tulip': 15,'water_lily': 16}
      
imgs = []
y = []

for flower in labels:
  for filename in glob.glob(f"/content/drive/MyDrive/archive/flowers/{flower}/*.jpg"): 
      #read in image as greyscale
      img = cv2.imread(filename, 0)

      # resize to 50x50
      img = cv2.resize(img, (75,75))
      y.append(labels[flower])
      imgs.append(img)
          
          
x_train, x_test, y_train, y_test = train_test_split(imgs, y, test_size=0.2)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

nsamples, nx, ny = x_train.shape
x_train2d = x_train.reshape((nsamples, nx*ny))

nsamples, nx, ny = x_test.shape
x_test2d = x_test.reshape((nsamples, nx*ny))

clf = svm.SVC()

clf.fit(x_train2d, y_train)

# Create predictions on the testing set
y_predict = clf.predict(x_test2d)

# compare the predicted labels to the actual labels and conver to percentage to output
acc = metrics.accuracy_score(y_test, y_predict)*100
print(f"{acc}% accuracy")

# saving the model to a pickle file
with open('model.pkl','wb') as f:
    pickle.dump(clf,f)

# ------------------------------------------- #
# example of loading the model
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
