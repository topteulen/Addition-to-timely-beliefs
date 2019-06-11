import csv
import datetime
import math

#input
#output rows of timely belief structure as array
def generator(csv_in,current_time,start_time,last_start_time,model):
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)[1:]
    index_current = get_row(current_time,data)
    index_start = get_row(start_time,data)
    index_last_start = get_row(index_last_start,data)
    if model == 'Tobias':
        data[index_start] = data[index_current]



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
    print("row on index position: ",data[index])
    print("index: ",index)
    return index

csv_file = 'temperature-linear_regressor-0.5.csv'
generator(csv_file,"2015-05-16 09:14:01+00:01","2015-05-20 09:14:00+00:00",0,'Tobias')
