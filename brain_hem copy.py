# Import statements
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Reading actual scan results 1 == hemorrhage present, 0 == not present
df = pd.read_csv('/Users/sulaimanjadallah/Desktop/capstone_2/labels.csv')
# Renaming columns
df = df.rename(columns={' hemorrhage': 'hemorrhage'})
# Fetching the CT scans creating a list of them
scans = glob.glob ('/Users/sulaimanjadallah/Desktop/capstone_2/head_ct' + '/*.png')

# Assigning images and labels to X and y
X = []
y = []

# Adding the true results to the y label (hem present or not present
for i in df.hemorrhage:
    y.append(i)

# Adding scans to X label

for scan in scans:
    study = cv2.imread(scan, cv2.IMREAD_COLOR)
    X.append(study)

# Changing datatypes to for model accessibility
X = np.array(X, dtype=object)
y = np.array(y)

# Splitting the data so that 67% of images and labels are used to train the data and 33%
# are used for testing the data

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.33)


# Resizing images for processing
def resize(data):
    resized = []
    for i in data:
        i = np.array(cv2.resize(i, dsize=(700, 600)))
        resized.append(i)
    return resized

train_data = resize(train_data)
test_data = resize(test_data)


plt.imshow(train_data[0])
plt.show()
plt.imshow(train_data[1])
plt.show()
#print(train_data[0].shape)
#print(train_data[1].shape)

train_data = np.array(train_data)
test_data = np.array(test_data)

print(train_data.shape)
print(test_data.shape)


train_data = train_data.reshape(134, 600, 700, 3)
train_data = train_data/255
test_data = test_data.reshape(66, 600, 700, 3)
test_data = test_data/255

print(train_data.shape, test_data.shape)

# Previous print statements confirm that the shapes of the imaging are all uniform

# Transforming training and testing labels to be categorical
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
labels = range(7)

num_classes = train_labels.shape[1]

# ML model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(600, 700, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_data, train_labels, epochs=11)
test_loss = model.evaluate(test_data, test_labels)

# Implimenting a confusion matrix to uncover type 1 and type 2 errors

y_true = []
for i in test_labels:
    if i[0] ==1:
        y_true.append(1)
    else:
        y_true.append(0)

# print(y_true)

y_pred = []
for i in model.predict(test_data):
    if i[0] > i[1]:
        y_pred.append(1)
    else:
        y_pred.append(0)
# print(y_pred)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

the_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(the_matrix, display_labels=['positive','negative' ])
disp.plot()
plt.show()
score = accuracy_score(y_true, y_pred)
print('Accuracy score is ' + str(score))