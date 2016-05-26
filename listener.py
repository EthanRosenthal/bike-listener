#!/home/ubuntu/miniconda2/bin/python
"""
Grab the current status of every citibike station. This script is to be run
on a cron for periodic station tracking.
"""

import pandas as pd
import json
import requests
import logging
from sqlalchemy import create_engine
import time
import numpy as np
import psycopg2
import yaml

LOG_FILENAME = 'listener.log'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.ERROR)

def load_config(conf_file):
    """Load database configuration file as global variables"""

    config = yaml.load(open(conf_file, 'r'))
    global DB_NAME
    DB_NAME = config['DB_NAME']
    global USER
    USER = config['USER']
    global PWD
    PWD = config['PWD']
    global TABLE
    TABLE = config['TABLE']
    global DB_TYPE
    DB_TYPE = config['DB_TYPE']

def mapColumns(cols, col_map):
    """
    Format column headers.

    Params
    ------
    cols : (list)
        List of column names for DataFrame
    col_map : (dict)
        Mapper to map column names from cols to new format

    Returns
    -------
    (list)
        List of newly formatted column names.
    """
    return [col_map[x] if col_map.has_key(x) else x for x in cols]

def get_status(api_url):
    """
    Make a GET request to the Citi Bike station api, convert to DataFrame and
    do a bit of cleaning.

    Params
    ------
    api_url : (str)
        url for GET request

    Returns
    -------
    stationdf : (DataFrame)
        Contains JSON text of GET Request as well as a repeated-value
        executionTime column.
    status_code : (int)
        Status code of GET request. 200 => Success
    """
    # Get station data and convert to DataFrame
    stations = requests.get(api_url)
    status_code = stations.status_code
    stations = json.loads(stations.text)
    stationlist = stations['stationBeanList']
    stationdf = pd.DataFrame(stationlist)

    # Make execution time series
    values = np.array([stations['executionTime']])
    values = np.repeat(values, stationdf.shape[0])

    ex = pd.DataFrame(index=np.arange(stationdf.shape[0]),
                      columns=['executionTime'])
    ex['executionTime'] = values

    cols = stationdf.columns.tolist() # Store for a hot minute
    stationdf = pd.concat([ex, stationdf], axis=1, ignore_index=True)
    stationdf.columns = ['executionTime'] + cols

    return stationdf, status_code

def write_status(api_url, engine, table):
    """
    Write the current status of all citibike docs to the database
    Params
    ------
    api_url : (str)
        url for citibike station GET request
    engine : (sqlalchemy engine)
        Connector to the database
    table : (str)
        Table to write the results to.
    """

    query = 'SELECT COALESCE(max(index),0)+1 FROM status'
    col_map = {'index':'index',
               'executionTime':'execution_time',
               'availableBikes':'available_bikes',
               'availableDocks':'available_docks',
               'id':'id',
               'lastCommunicationTime':'last_communication_time',
               'latitude':'lat',
               'longitude':'lon',
               'stAddress1':'st_address',
               'stationName':'station_name',
               'statusKey':'status_key',
               'statusValue':'status_value',
               'testStation':'test_station',
               'totalDocks':'total_docks'}
    post_cols = ['index',
                 'execution_time',
                 'available_bikes',
                 'available_docks',
                 'id',
                 'last_communication_time',
                 'lat',
                 'lon',
                 'st_address',
                 'station_name',
                 'status_key',
                 'status_value',
                 'test_station',
                 'total_docks']

    print '{}: write_status()'.format(time.ctime())

    try:
        df, status = get_status(api_url)
        if status != 200:
            logging.error('Error: bad request at : {}'.format(time.ctime()))
        else:
            df.columns = mapColumns(df.columns.tolist(), col_map)
            df['index'] = pd.read_sql_query(query, engine).iloc[0,0] + \
                          range(len(df))
            df[post_cols].to_sql(table,
                                 engine,
                                 if_exists='append',
                                 index=False)
    except:
        logging.error('Error: bad request at : {}'.format(time.ctime()))


def create_table(db, user, pwd, table):
    # host = '*' needed for some ec2 postgres reason I don't understand
    conn = psycopg2.connect(dbname=db, user=user, password=pwd, host='*')
    query = """
            CREATE TABLE {table} (
            index bigserial NOT NULL,
            execution_time TIMESTAMP,
            available_bikes SMALLINT,
            available_docks SMALLINT,
            id SMALLINT,
            last_communication_time TIMESTAMP,
            lat NUMERIC(8, 6),
            lon NUMERIC(8, 6),
            st_address VARCHAR(100),
            station_name VARCHAR(100),
            status_key SMALLINT,
            status_value VARCHAR(100),
            test_station BOOLEAN,
            total_docks SMALLINT
            );
            """.format(**{'table':table})
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':

    load_config(sys.argv[1])

    engine_str = '{}://{}:{}@localhost/{}'.format(DB_TYPE, USER, PWD, DB_NAME)
    engine = create_engine(engine_str)

    write_status(API_URL, engine, TABLE)
