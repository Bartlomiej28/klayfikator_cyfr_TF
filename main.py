import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

plt.imshow(X_train[50], cmap='gray')
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

loss, accuracy = model.evaluate(X_test, y_test)
print("Dokładność:", accuracy)

plt.imshow(X_test[0], cmap='gray')
plt.show()
print("Etykieta rzeczywista:", y_test[0])

y_pred = model.predict(X_test)
print("Wektor wyjściowy (prawdopodobieństwa):", y_pred[0])

predicted_label = np.argmax(y_pred[0])
print("Etykieta przewidziana:", predicted_label)

y_pred_labels = [np.argmax(i) for i in y_pred]

conf_mat = confusion_matrix(y_test, y_pred_labels)
print("Macierz pomyłek:\n", conf_mat)

plt.figure(figsize=(15, 7))
sns.heatmap(conf_mat.numpy(), annot=True, fmt='d', cmap='Blues')
plt.ylabel('Etykiety rzeczywiste')
plt.xlabel('Etykiety przewidziane')
plt.show()
