from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly
from bokeh.palettes import cividis, viridis
import colorcet as cc
import datetime
import csv
import math
import numpy as np
import scipy.stats as stats
import pandas as pd


def ridgeline_plot(date, csv_005, csv_05, csv_095, output=False, start=0, end=168):
    """ 
    Creates rigdeline plot 

    @param date : datetime string
    @param csv_005 : input csv file with 5 percent probability
    @param csv_05 : input csv file with 50 percent probability
    @param csv_095 : input csv file with 95 percent probability
    @param output : if true write to output png file
    @param interval : to be added
    """
    start_index = start + 2
    end_index = end + 2
    data_05 = create_data(csv_05)
    data_095 = create_data(csv_095)
    index_05 = get_row(date, data_05)
    index_095 = get_row(date, data_095)
    pred_temp_05 = data_05[index_05][start_index:end_index]
    pred_temp_095 = data_095[index_095][start_index:end_index]
    mean = np.array([float(i) for i in pred_temp_05])
    sigma = np.array([float(i) / 4 for i in pred_temp_095])

    nr_lines = end_index - start_index

    show_plot(mean, sigma, nr_lines)
    return mean, sigma


def show_plot(mean, sigma, nr_lines):
    """
    Creates and shows ridgline plot
    
    @param mean : list of mean values
    @param sigmaa : list of sigma values
    @param nr_lines : number of pdfs to be drawn
    """
    x = np.linspace(-10, 70, 500)
    frame = pd.DataFrame()
    for i in range(nr_lines):
        frame["{}".format(i)] = stats.norm.pdf(x, mean[i], sigma[i])

    cats = list(reversed(frame.keys()))
    pallete = viridis(nr_lines)
    source = ColumnDataSource(data=dict(x=x))

    p = figure(y_range=cats, plot_width=900, x_range=(-5, 60), toolbar_location=None)

    for i, cat in enumerate(reversed(cats)):
        y = ridge(cat, frame[cat])
        source.add(y, cat)
        p.patch('x', cat, alpha=0.6, color=pallete[i], line_color="black", source=source)

    p.outline_line_color = None
    p.background_fill_color = "#ffffff"

    p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#000000"
    p.xgrid.ticker = p.xaxis[0].ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None
    p.yaxis.visible = None
    p.y_range.range_padding = 0.2

    show(p)


def ridge(category, data, scale=100):
    return list(zip([category] * len(data), scale * data))


def get_row(current_time, data):
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
            R = index - 1
        else:
            L = index + 1
    return index


def create_data(csv_in):
    """
    Returns data as list type

    @param csv_in : input csv file  
    """
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)
    return data


# ridgeline_plot('2015-06-16 10:30:00', 'temperature-random_forest-0.05.csv', 'temperature-random_forest-0.5.csv', 'temperature-random_forest-0.95.csv')
