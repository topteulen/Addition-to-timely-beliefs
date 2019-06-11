import csv
import datetime
import pytz
import math
import numpy as np
import timely_beliefs as tb
from datetime import timedelta
from timely_beliefs.beliefs.utils import load_time_series
import pandas as pd
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
        return model.predict(np.array([[observation]]))[0][0]
    print("Other model functionality has not yet been implemented")
    return observation

#input
#output rows of timely belief structure as array
def main(csv_in,current_time,start_time,last_start_time,model, addtocsv = False):
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        datacomp = list(csv_reader)
        data = datacomp[1:]
        print(len(data))
    index_current = get_row(current_time,data)
    index_start = get_row(start_time,data)
    index_last_start = get_row(last_start_time,data)
    if model == 'Tobias':
        with open('temp.csv', 'w') as csvfile0:
            writer = csv.writer(csvfile0, delimiter=',')
            index_list = range(index_start,index_last_start+1)
            result = []
            for index in index_list:
                value = generator(csv_in, current_time)
                result += [[data[index][0],data[index_current][0],'Test',0.5,value]]
                if addtocsv == True:
                    print((index-index_current)*0.15)
                    datacomp[index][round((index-index_current)*0.15)] = value
            columns = ["event_start","belief_time","source","cumulative_probability","event_value"]
            writer.writerow(columns)
            writer.writerows(result)
            print(datacomp[index][round((index-index_current)*0.15)])
        t = pd.read_csv("temp.csv",index_col=0)
        if addtocsv == True:
            with open(csv_in, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerows(datacomp)
    return t


#input current_time &datetime string
#input data &list of data
def get_row(current_time,data):
    #convert string to datetime object for comparing
    datetime_object = datetime.datetime.strptime(current_time,'%Y-%m-%d %H:%M:%S%z')
    #set left and right halfs
    L = 0
    R = len(data)-1
    lastindex = 0
    while(L <= R):
        #set middle point/ search index
        index = math.floor((L+R)/2)
        #round to closest value if exact value not found
        if index == lastindex:
            break
        lastindex = index
        #if time found return
        if datetime.datetime.strptime(data[index][0],'%Y-%m-%d %H:%M:%S%z') == datetime_object:
            break
        elif datetime.datetime.strptime(data[index][0],'%Y-%m-%d %H:%M:%S%z') > datetime_object:
            R = index - 1
        else:
            L = index + 1
    #print("row on index position: ",data[index])
    #print("index: ",index)
    return index

csv_file = 'temperature-linear_regressor-0.5.csv'
main(csv_file,"2015-05-16 09:14:01+00:01","2015-05-20 09:14:00+00:00","2015-05-20 09:30:15+00:00",'Tobias',addtocsv=True)
