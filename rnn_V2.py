import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,RNN, Input,Embedding
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from matplotlib import pyplot
from datetime import datetime,timedelta
#data = np.loadtxt("energy_data.csv",delimiter=',')
#X=data[:,0:8]
#Y=data[:,8]
history_size = 5

with open('energy_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    data = list(csv_reader)[1:]

data = np.array(data)
temps = data[:,-1]
maxtemp = 0
for temp in temps:
    if float(temp) > maxtemp:
        maxtemp = float(temp)
print(maxtemp)
print(datetime.strptime(data[0][0][:16],'%Y-%m-%d %H:%M'))
newdata = []
for i in range(len(data)):
    time = datetime.strptime(data[i][0][:16],'%Y-%m-%d %H:%M') - datetime(2010,1,1,0,0)
    data[i][0] = time.total_seconds()/3600
    if i >= history_size+1:
        newdata += [[]]
        newdata[i-(history_size+1)] += [float(data[i-((history_size+1)-j)][-1])/maxtemp  for j in range(history_size)]


data = np.array(newdata)
print(data.shape)
num_temps = len(newdata)
label_array = np.zeros((len(newdata),int(len(newdata)*0.7)),dtype=np.int8)

for feature_index,temp_index in enumerate(temps):
    if feature_index < len(newdata)-1:
        index = int(float(temp_index)*10)
        label_array[feature_index,index] = 1

X = data
Y = label_array
train_x = X[:int(len(X)*0.7)]
test_x =  X[int(len(X)*0.7)+1:]
train_y = Y[:int(len(Y)*0.7)]
test_y =  Y[int(len(Y)*0.7)+1:]
train_x = train_x.reshape((train_x.shape[0],1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0],1,test_x.shape[1]))
print(train_x.shape)
print(train_y.shape)


model = Sequential()
# design network

# Recurrent layer
model.add(LSTM(64, return_sequences=True,
               dropout=0.1, recurrent_dropout=0.1))

model.add(LSTM(64, return_sequences=False,recurrent_dropout=0.1))
# Fully connected layer
model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(train_x.shape[0], activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit network
history = model.fit(train_x, train_y, epochs=20, batch_size=30, validation_data=(test_x, test_y), verbose=1, shuffle=True)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
