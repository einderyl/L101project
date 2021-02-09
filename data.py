# -*- coding: utf-8 -*-

import pandas as pd
import requests
import json
import csv
import time
from datetime import datetime, timedelta

def date_to_time(datetime_obj):
    return int(datetime_obj.timestamp())

def filter(data, threshold=100):
    indices = [i for i in range(len(data)) if len(data[i].get('selftext', '')) > threshold]
    return [data[i] for i in indices]

def req_data(subreddit): 
    data = []
    url = r'https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size=100'
    first_day = datetime(year=2017, month=1, day=1)
    delta = timedelta(days=1)

    day = 0
    
    while day < 365*3:
        start = date_to_time(first_day)
        first_day += delta
        end = date_to_time(first_day)

        try:
          response = requests.get(url.format(subreddit, start, end))
        except:
            print(f'retrying day {day}')
            time.sleep(60)

        if response.status_code == 200:
            data += filter(json.loads(response.content)['data'])
            day += 1
        else:
            print(f'retrying day {day}')
            time.sleep(60)
        
    return data

d = req_data('singapore')
df = pd.DataFrame(d)
df.to_csv('sg.csv')

d = req_data('UnitedKingdom')
df = pd.DataFrame(d)
df.to_csv('uk.csv')

