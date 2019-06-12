import csv
import datetime
import math
import numpy as np
from sklearn.linear_model import LinearRegression

def get_observation(csv_in,current_time):
    """
    Returns observation at given datetime

    @param csv_in : csv file containing all forecast data
    @param current_time : datetime string to observe
    @param model : model to use to generate new data 
    """
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)[1:]
    index = get_row(current_time,data)
    return float(data[index][1])

def generator(csv_in, current_time, model=LinearRegression()):
    """
    Generates a new forecast at a given time using a model

    @param csv_in : csv file containing all forecast data
    @param current_time : datetime string to observe
    @param model : model to use to generate new data 
    """
    observation = get_observation(csv_in, current_time)
    if str(type(model)) == "<class 'sklearn.linear_model.base.LinearRegression'>":
        X = np.array([[1], [2], [3], [4]]) # placeholder
        model.fit(X,X)
        return model.predict(np.array([[observation]]))[0]
    print("Other model functionality has not yet been implemented")
    return list(observation)


#input current_time &datetime string
#input data &list of data
def get_row(current_time ,data):
    """
    Returns index of current_time in data

    @param current_time : datetime string to be searched
    @param data : read csv file
    """
    datetime_object = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    L = 0
    R = len(data) - 1
    lastindex = 0
    while(True):
        index = math.floor((L + R) / 2)
        if index == lastindex:
            break
        lastindex = index
        if datetime.datetime.strptime(data[index][0][:-6], '%Y-%m-%d %H:%M:%S') == datetime_object:
            break
        elif datetime.datetime.strptime(data[index][0][:-6], '%Y-%m-%d %H:%M:%S') > datetime_object:
            if index > 0:
                R = index - 1
        else:
            L = index + 1
    return index

# testing
csv_file = 'temperature-linear_regressor-0.5.csv'
print(generator(csv_file,"2015-06-01 00:00:00"))
