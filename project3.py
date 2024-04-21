import numpy as np

data = np.loadtxt("dl.csv", delimiter=",")
x = data[:,0:8]
y = data[:,8]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x,y,epochs=300, batch_size=10)