import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,RNN, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from matplotlib import pyplot
from datetime import datetime,timedelta
#data = np.loadtxt("energy_data.csv",delimiter=',')
#X=data[:,0:8]
#Y=data[:,8]

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
    if i >= 101:
        newdata += [[]]
        newdata[i-101] += [float(data[i-(101-j)][-1])/maxtemp for j in range(100)]


data = np.array(newdata)
X = data[:,0:101]
Y = data[:,5]
train_x = X[:int(len(X)*0.7)]
test_x =  X[int(len(X)*0.7)+1:]
train_y = Y[:int(len(Y)*0.7)]
test_y =  Y[int(len(Y)*0.7)+1:]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape)
print(test_y.shape)



# design network
model = Sequential()
model.add(RNN(200))
model.add(RNN(200))
model.add(Dense(1, activation='relu'))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
# fit network
history = model.fit(train_x, train_y, epochs=20, batch_size=100, validation_data=(test_x, test_y), verbose=1, shuffle=True)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()



# print(X)
# print(Y)
# random_df = df.sample(frac=1)
# train = random_df[:int(len(random_df)*0.7)]
# test =  random_df[int(len(random_df)*0.7)+1:]
#
# train =
#
# print(trainX)
# print(y_train)
# lb = preprocessing.LabelBinarizer()
# y_train = lb.fit_transform(y_train)
# y_test = lb.fit_transform(y_test)
# print(y_train)
# print(y_test)
#
# def load_data():


# model = Sequential()
# model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(LSTM(hidden_size, return_sequences=True))
# if use_dropout:
#     model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(vocabulary)))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
#
# checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
#
#
# model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
#                         validation_data=valid_data_generator.generate(),
#                         validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
#

#
#
# model = load_model(data_path + "\model-40.hdf5")
# dummy_iters = 40
# example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
#                                                      skip_step=1)
# print("Training data:")
# for i in range(dummy_iters):
#     dummy = next(example_training_generator.generate())
# num_predict = 10
# true_print_out = "Actual words: "
# pred_print_out = "Predicted words: "
# for i in range(num_predict):
#     data = next(example_training_generator.generate())
#     prediction = model.predict(data[0])
#     predict_word = np.argmax(prediction[:, num_steps-1, :])
#     true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
#     pred_print_out += reversed_dictionary[predict_word] + " "
# print(true_print_out)
# print(pred_print_out)
#
# num_steps=30
# batch_size=20
# hidden_size=500
