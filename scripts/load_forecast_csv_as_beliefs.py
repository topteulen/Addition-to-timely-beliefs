#!/usr/bin/env python

from datetime import datetime, timedelta
import sys
import time
import pickle

import pytz
import isodate
import pandas as pd
import timely_beliefs as tb
from timely_beliefs.beliefs.utils import load_time_series


if len(sys.argv) <= 1:
    print("Please use either 'plot' or 'pickle' as parameter! (Use `plot serve open` to plot directly in your browser.")
    sys.exit(2)

sensor_descriptions = (
    ("Solar irradiation", "kW/m²"),
    ("Solar power", "kW"),
    ("Wind speed", "m/s"),
    ("Wind power", "MW"),
)

n_events = 4 * 24# * 7  # For playing around fast we will read in beliefs only about the first n events
n_horizons = 20
show_accuracy = True
active_fixed_viewpoint_selector = True
tz_hour_difference = -9
event_resolution = timedelta(hours=1)

cols = [0, 1]  # Columns with datetime index and observed values
horizons = list(range(0, 4, 1)) + list(range(4, 6, 2)) + list(range(6, 12, 3)) + list(range(12, 24, 4)) + list(range(24, 36, 6)) + list(range(36, 2*24, 12)) + list(range(2*24, 3*24, 24)) #  + list(range(3*24, 7*24, 48))  # + [7*24 - 1]
cols.extend([h + 2 for h in horizons])


class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()
        print("[%s] Starting (at %s) ..." % (self.name, datetime.fromtimestamp(self.tstart).strftime("%A, %B %d, %Y %I:%M:%S")))

    def __exit__(self, type, value, traceback):
        duration = time.time() - self.tstart
        if duration > 1:
            message = "Elapsed: %.2f seconds" % duration
        else:
            message = "Elapsed: %.0f ms" % (duration * 1000)
        if self.name:
            message = "[%s] " % self.name + message
        print(message)
        if self.filename:
            with open(self.filename, "a") as file:
                print(str(datetime.now()) + ": ", message, file=file)


def read_beliefs_from_csv(sensor, source, cp, event_resolution: timedelta, tz_hour_difference: float = 0) -> list:
    beliefs = pd.read_csv("../%s-%s-%s.csv" % (sensor.name.replace(' ', '_').lower(), source.name.replace(' ', '_').lower(), cp),
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


source_a = tb.BeliefSource(name="Linear regression")
source_b = tb.BeliefSource(name="XGBoost")
source_c = tb.BeliefSource(name="Random forest")

sensors = (tb.Sensor(name=descr[0], unit=descr[1], event_resolution=event_resolution) for descr in sensor_descriptions)

for sensor in sensors:
    blfs = read_beliefs_from_csv(sensor, source=source_a, cp=0.5, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
    blfs += read_beliefs_from_csv(sensor, source=source_b, cp=0.5, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
    blfs += read_beliefs_from_csv(sensor, source=source_c, cp=0.05, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
    blfs += read_beliefs_from_csv(sensor, source=source_c, cp=0.5, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
    blfs += read_beliefs_from_csv(sensor, source=source_c, cp=0.95, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)

    bdf = tb.BeliefsDataFrame(sensor=sensor, beliefs=blfs).sort_index()
    print("----------------------------------")
    print("Made BeliefsDataFrame for sensor %s:" % sensor.name)
    print(bdf.info())
    print("----------------------------------")

    if sys.argv[1] == "plot":
        with Timer("Time to plot"):
            chart = bdf.plot(show_accuracy=show_accuracy, active_fixed_viewpoint_selector=active_fixed_viewpoint_selector, reference_source=bdf.lineage.sources[0])
        if len(sys.argv) > 2 and sys.argv[2] == "serve":
            if len(sys.argv) > 3 and sys.argv[3] == "open":
                chart.serve(open_browser=True)
            else:
                chart.serve(open_browser=False)
        else:
            chart.save("%s.chart.json" % sensor.name.replace(" ", "_").lower())
    elif sys.argv[1] == "pickle":
        with open("%s.pickle" % sensor.name.replace(" ", "_").lower(), "wb") as df_file:
            pickle.dump(bdf, df_file)




# # Verify that the fixed viewpoint corresponds to the diagonal in the csv
# print(bdf.fixed_viewpoint(datetime(2014, 12, 31, 15, 45, tzinfo=pytz.utc)))
#
# # Verify that the belief history corresponds to the horizontal in the csv
# print(bdf.belief_history(datetime(2014, 12, 31, 16, 45, tzinfo=pytz.utc)))
#
# # Verify that the rolling viewpoint corresponds to the vertical in the csv
# print(bdf.rolling_viewpoint(timedelta(hours=1)).dropna())
#
# # Check out accuracy metrics versus forecast horizon
# print(bdf.accuracy())
