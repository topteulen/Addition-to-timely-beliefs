from datetime import datetime, timedelta
from timely_beliefs.beliefs.utils import load_time_series
from scipy.special import erfinv
from bokeh.palettes import viridis
from bokeh.io import show, export_png
from bokeh.models import ColumnDataSource, FixedTicker, FuncTickFormatter
from bokeh.models import SingleIntervalTicker, LinearAxis
from bokeh.plotting import figure
import time, pickle
import datetime
import pytz
import isodate
import pandas as pd
import timely_beliefs as tb
import numpy as np
import scipy.stats as stats


def read_beliefs_from_csv(sensor, source, cp, event_resolution: timedelta, tz_hour_difference: float = 0) -> list:
    """
    Returns a timely_beliefs DataFrame read from a csv file
    
    @param sensor : sensor used
    @param source : type of model used
    @param cp : cummulative probability
    @param event_resolution : event resolution in timedelta hours
    @param tz_hour_difference : time difference 
    """
    sensor_descriptions = (("Temperature", "C"),)

    cols = [0, 1]  # Columns with datetime index and observed values
    horizons = list(range(0, 169, 1)) 
    cols.extend([h + 2 for h in horizons])
    n_horizons = 169
    n_events = None
    beliefs = pd.read_csv("%s-%s-%s.csv" % (sensor.name.replace(' ', '_').lower(), source.name.replace(' ', '_').lower(), cp),
                          index_col=0, parse_dates=[0], date_parser=lambda col: pd.to_datetime(col, utc=True) - timedelta(hours=tz_hour_difference),
                          nrows=n_events, usecols=cols)
    beliefs = beliefs.resample(event_resolution).mean()
    assert beliefs.index.tzinfo == pytz.utc
    # Construct the BeliefsDataFrame by looping over the belief horizons
    blfs = load_time_series(beliefs.iloc[:, 0].head(n_events), sensor=sensor, source=source,
                            belief_horizon=timedelta(hours=0), cumulative_probability=0.5)  # load the observations (keep cp=0.5)
    for h in beliefs.iloc[:, 1 :n_horizons + 1] :
        try:
            blfs += load_time_series(beliefs[h].head(n_events), sensor=sensor, source=source,
                                     belief_horizon=(isodate.parse_duration(
                                     "PT%s" % h)) + event_resolution, cumulative_probability=cp)  # load the forecasts
        except isodate.isoerror.ISO8601Error:  # In case of old headers that don't yet follow the ISO 8601 standard
            blfs += load_time_series(beliefs[h].head(n_events), sensor=sensor, source=source,
                                     belief_horizon=(isodate.parse_duration(
                                     "%s" % h)) + event_resolution, cumulative_probability=cp)  # load the forecasts
    return blfs

def make_df(n_events = 100 , n_horizons = 169, tz_hour_difference=-9, event_resolution= timedelta(hours=1)):
    """
    Returns DataFrame
    
    @param n_events : number of events in DataFrame
    @param n_horizons : number of horizons in DataFrame
    @param tz_hour_difference : time difference 
    @param event_resolution : event resolution in timedelta hours
    """
    sensor_descriptions = (("Temperature", "C"),)

    source = tb.BeliefSource(name="Random forest")

    sensors = (tb.Sensor(name=descr[0], unit=descr[1], event_resolution=event_resolution) for descr in sensor_descriptions)
    blfs=[]
    for sensor in sensors:
        blfs += read_beliefs_from_csv(sensor, source=source, cp=0.05, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
        blfs += read_beliefs_from_csv(sensor, source=source, cp=0.5, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
        blfs += read_beliefs_from_csv(sensor, source=source, cp=0.95, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)

        bdf = tb.BeliefsDataFrame(sensor=sensor, beliefs=blfs).sort_index()

    return bdf

def create_cp_data(df, first_belief_time, last_belief_time, start_time):
    """
    Returns lists with values of 0.05, 0.5 and 0.95 cumulative probability
    
    @param df : DataFrame containing events, belief times, predictions and their cumulative probability
    @param first_belief_time : first belief time 
    @param last_belief_time : last belief time 
    @param start_time : time of event
    """
    temp_time = first_belief_time
    list_005 = []
    list_05 = []
    list_095 = []
    while temp_time <= last_belief_time:
        list_005  +=  [get_beliefsSeries_from_event_start(df, start_time, temp_time)[0]]
        list_05 +=  [get_beliefsSeries_from_event_start(df, start_time, temp_time)[1]]
        list_095 +=  [get_beliefsSeries_from_event_start(df, start_time, temp_time)[2]]
        length = len(list_005)
        if any(len(lst) != length for lst in [list_005, list_05, list_095]):
            raise StandardError("Could not find all cumulative probability values")
        temp_time += df.sensor.event_resolution

    return (list_005, list_05, list_095)


def get_beliefsSeries_from_event_start(df, datetime_object,current_time):
    return df.loc[(datetime_object.strftime("%m/%d/%Y, %H:%M:%S"),current_time.strftime("%m/%d/%Y, %H:%M:%S")),'event_value']



def ridgeline_plot(date, df, start=0, end=168):
    """ 
    Creates ridgeline plot by selecting a belief history about a specific event

    @param date : datetime string of selected event
    @param df : timely_beliefs DataFrame
    @param start : start of hours before event time
    @param end : end of hours before event time
    """
    if end < 0 or end > 168:
        raise ValueError("End of the forecast horizon must be between 0 and 168 hours.")
    if start < 0 or start > end:
        raise ValueError("Start of the forecast horizon must be between 0 and 168 hours.")

    start_time = date
    first_belief_time = date - ((end-1)*df.sensor.event_resolution)
    last_belief_time = date - ((start)*df.sensor.event_resolution)

    pred_temp_005, pred_temp_05, pred_temp_095 = create_cp_data(df,first_belief_time,last_belief_time,start_time)
    
    mean = np.array([float(i) for i in pred_temp_05])
    sigma1 = np.array([(float(pred_temp_095[i])-float(pred_temp_05[i]))/(np.sqrt(2)*erfinv((2*0.95)-1)) for i in range(len(pred_temp_05))])
    sigma2 = np.array([(float(pred_temp_005[i])-float(pred_temp_05[i]))/(np.sqrt(2)*erfinv((2*0.05)-1)) for i in range(len(pred_temp_05))])
    sigma = (sigma1+sigma2)/2

    show_plot(mean, sigma, start, end)


def show_plot(mean, sigma, start, end):
    """
    Creates and shows ridgeline plot

    @param mean: list of mean values
    @param sigma: list of sigma values
    @param start: start hours before event-time
    @param end: end hours before event-time
    """
    nr_lines = end - start
    x = np.linspace(-10, 30, 500)
    frame = pd.DataFrame()
    for i in range(nr_lines):
        frame["{}".format(i)] = stats.norm.pdf(x, mean[i], sigma[i])
    cats = list(reversed(frame.keys()))
    pallete = viridis(nr_lines)
    source = ColumnDataSource(data=dict(x=x))

    p = figure(y_range=cats, plot_width=900, x_range=(-5, 30), toolbar_location=None)

    for i, cat in enumerate(reversed(cats)):
        y = ridge(cat, frame[cat], 50)
        source.add(y, cat)
        p.patch('x', cat, alpha=0.6, color=pallete[i], line_color="black", source=source)
        

    p.outline_line_color = None
    p.background_fill_color = "#ffffff"

    p.xaxis.ticker = FixedTicker(ticks=list(range(-20, 101, 10)))
    p.xaxis.axis_label = 'Temperature (Celcius)'
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#000000"
    p.xgrid.ticker = p.xaxis[0].ticker
    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = 0.2 / (nr_lines / 168)
    
    p.yaxis.axis_label = 'Number of hours before event-time'
    y_ticks = list(np.arange(end, 0, -5)) 
    yaxis = LinearAxis(ticker=y_ticks)
    
    y_labels = list(np.arange(start, end, 5))
    mapping_dict = {y_ticks[i]: str(y_labels[i]) for i in range(len(y_labels))}
    for i in range(end+1):
        if i not in mapping_dict:
            mapping_dict[i]=" "
    mapping_code = "var mapping = {};\n    return mapping[tick];\n    ".format(mapping_dict)
    p.yaxis.formatter = FuncTickFormatter(code=mapping_code)    
    show(p)


def ridge(category, data, scale=100):
    return list(zip([category] * len(data), scale * data))

# df = make_df()
# ridgeline_plot(datetime.datetime(2015, 3, 1, 9, 0, tzinfo=pytz.utc), df)

