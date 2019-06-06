from datetime import timedelta

import pytz
import pandas as pd
import timely_beliefs as tb
from timely_beliefs.beliefs.utils import load_time_series


n_events = 4 * 24# * 7  # For playing around fast we will read in beliefs only about the first n events (or set to None)
tz_hour_difference = -9  # For sake of simplicity, let's pretend the data was measured in a UTC timezone

sensor_descriptions = (
    ("Solar irradiation", "kW/m²"),
    ("Solar power", "kW"),
    ("Wind speed", "m/s"),
    ("Wind power", "MW"),
    ("Temperature", "°C"),
)


def read_beliefs_from_csv(sensor, source, event_resolution: timedelta = None, tz_hour_difference: float = 0, n_events: int = None) -> list:
    beliefs = pd.read_csv("../energy_data.csv",
                          index_col=0, parse_dates=[0],
                          date_parser=lambda col : pd.to_datetime(col, utc=True) - timedelta(hours=tz_hour_difference),
                          nrows=n_events, usecols=["datetime", sensor.name.replace(' ', '_').lower()])
    if event_resolution is not None:
        beliefs = beliefs.resample(event_resolution).mean()
    assert beliefs.index.tzinfo == pytz.utc

    # Construct the BeliefsDataFrame by looping over the belief horizons
    blfs = load_time_series(beliefs[sensor.name.replace(' ', '_').lower()], sensor=sensor, source=source,
                            belief_horizon=timedelta(hours=0), cumulative_probability=0.5)  # load the observations (keep cp=0.5)

    return blfs


# Create source and sensors
source_a = tb.BeliefSource(name="KNMI")
sensors = (tb.Sensor(name=descr[0], unit=descr[1], event_resolution=timedelta(minutes=15)) for descr in sensor_descriptions)

# Create BeliefsDataFrame
for sensor in sensors:
    blfs = read_beliefs_from_csv(sensor, source=source_a, tz_hour_difference=tz_hour_difference, n_events=n_events)
    df = tb.BeliefsDataFrame(sensor=sensor, beliefs=blfs).sort_index()
    print(df)
