import requests
import datetime
import pandas as pd

def get_temperatures(lat,lon,start_date,end_date,):
    """
    Parameters
    ----------
    latitude : float
    longitude : float
    start_date : datetime object
    end_date : datetime object

    Returns
    -------
    pandas.DataFrame
    """

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature_2m": data["hourly"]["temperature_2m"],
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df['time'] >= start_date) & (df['time'] < end_date)]

    return df