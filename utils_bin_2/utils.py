import pandas as pd
import numpy as np
import geopandas as geopd
from sklearn.preprocessing import MinMaxScaler
from math import radians, cos, sin, asin, sqrt
from scipy import stats

def time_shifter(string):
  """Shifts time so that it is defined from 00 to 23

  Args:
      string (str): Input datetime as string

  Returns:
      str: Returns a correct version of the datetime string
  """
  if string[:2] == "24":
    return "00"+string[2:]
  if string[:2] == "25":
    return "01"+string[2:]
  return string

def index_transformer(input_df, resample = "60min"):
  """Transforming index to pd.DatetimeIndex and resample

  Args:
      input_df (pd.DataFrame)
      resample (str, optional): Resample frequency. Defaults to "60min".

  Returns:
      pd.DataFrame
  """
  input_df.index = pd.to_datetime(input_df.index)
  input_df = input_df.sort_index()
  return input_df.resample(resample).sum()

def extract_stops_resample(input_df, resample = "60min"):
  """Transforming index to pd.DatetimeIndex and resample without losing columns

  Args:
      input_df (pd.DataFrame)
      resample (str, optional): Resample frequency. Defaults to "60min".

  Returns:
      pd.DataFrame
  """

  column = input_df.loc[:,input_df.dtypes == "object"].columns[0]
  stop_id = input_df.loc[:,input_df.dtypes == "object"].values.flatten()[0]
  
  resampled_df = input_df.resample(resample).sum()

  correct_ind_df = pd.DataFrame(index = pd.date_range(resampled_df.index[0], \
    periods = 24, freq = "60T"))
  
  correct_ind_df = correct_ind_df.join(resampled_df, how = "left").fillna(0)
  correct_ind_df.index = correct_ind_df.index.time
  correct_ind_df = correct_ind_df.sort_index()

  correct_ind_df[column] = stop_id
  return correct_ind_df[input_df.columns]


def string_comparator(input_df, column_1, column_2):
  """Simple comparison the overlap between the overlap of two strings

  Args:
      input_df (pd.DataFrame)
      column_1 (pd.Series): Series composed by set of strings
      column_2 (pd.Series): Series composed by set of strings

  Returns:
      pd.DataFrame: Values are a similar index between the the rows and columns,
      which are composed by the strings of column_1 and column_2, respectively
  """
  list_elements_1 = input_df[column_1].str.split().values
  list_elements_2 = input_df[column_2].str.split().values

  final_df = pd.DataFrame(index = input_df[column_1].values, columns = input_df[column_2].values)

  for ind, string_list_1 in enumerate(list_elements_1):
      for col, string_list_2 in enumerate(list_elements_2):
          
          if len(string_list_1) >= len(string_list_2):
              norm = len(string_list_1)
          else:
              norm = len(string_list_2)
          
          overlap = len([ele for ele in string_list_1 if ele in string_list_2])

          final_df.iloc[ind, col] = overlap/norm
          
  return final_df

def mercator_transform(input_df):
  """Generates GeoPandas DataFrame in EPSG:3857 map format 

  Args:
      input_df (geopd)

  Returns:
      geopd
  """

  long_col = [ele for ele in input_df.columns if "lon" in ele][0]
  lat_col = [ele for ele in input_df.columns if "lat" in ele][0]
  
  positions_df = geopd.GeoDataFrame(crs="EPSG:4326", geometry = geopd.points_from_xy(x=input_df[long_col]\
    , y=input_df[lat_col]), index = input_df.index)
  
  return positions_df.to_crs("EPSG:3857")

def haversine(lon1, lat1, lon2, lat2):
  """  Calculate the great circle distance in kilometers between two points 
  on the earth (specified in decimal degrees)

  Args:
      lon1 (float): Longitude of point 1 in degrees
      lat1 (float): Latitude of point 1 in degrees
      lon2 (float): Longitude of point 2 in degrees
      lat2 (float): Longitude of point 2 in degrees

  Returns:
      float: Distance in kilometers between points 1 and 2
  """
  # convert decimal degrees to radians 
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

  # haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a)) 
  r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
  return c * r

def stop_amenity_weighter(stops_df, amenities_df, threshold = 2, reg = 0.2):
  """  Computes the weight of a stop given the proximity (in Km) and the type of amenity
  Return a dictionary where the keys are the stops and the values are the 
  weight importance of the amenities

  Choices for the weight function are:
  1. Linear in the amenity weights and inversely proportional to the (regularized distance);
  2. Using Burr distribution.

  Args:
      stops_df (pd.DataFrame): Contains geographical locations of certain reference points (in particular, bus stops).
      amenities_df (pd.DataFrame): Contains geographical locations of certain relevant venues (for example, cinemas).
      threshold (int, optional): Radius below which amenities are included in the final score. Defaults to 2.
      reg (float, optional): Regularization factor. Necessary when using weight function 1. Defaults to 0.2.

  Returns:
      dict: Keys are the reference point id and the values are the weights for each amenity.
  """
  distance_weight_dict = {}
  amenities_weight_dict = {'fast_food':1, 'library':2, 'school':2, "theatre":2, 'cinema': 2,'hospital':3, 'college':5,
       'university':6} ##This should be defined as an input

  stop_long_col = [ele for ele in stops_df.columns if "lon" in ele][0]
  stop_lat_col = [ele for ele in stops_df.columns if "lat" in ele][0]

  amen_long_col = [ele for ele in amenities_df.columns if "lon" in ele][0]
  amen_lat_col = [ele for ele in amenities_df.columns if "lat" in ele][0]
  amen_tags_col = [ele for ele in amenities_df.columns if "tags" in ele][0]

  for stop_ind in stops_df.index:

      stop_long = stops_df.loc[stop_ind, stop_long_col]
      stop_lat = stops_df.loc[stop_ind, stop_lat_col]

      distance_weight_dict[stop_ind] = []

      for amenity_ind in amenities_df.index:
          
          amen_long = amenities_df.loc[amenity_ind, amen_long_col]
          amen_lat = amenities_df.loc[amenity_ind, amen_lat_col]
          amen_tag = amenities_df.loc[amenity_ind, amen_tags_col]

          stop_amen_dist = haversine(stop_long, stop_lat, amen_long, amen_lat)
          amenity_weight = amenities_weight_dict[amen_tag]

          if stop_amen_dist < threshold:
              #weight_function = amenity_weight/(reg + stop_amen_dist)
              weight_function = amenity_weight*stats.burr.pdf(stop_amen_dist, 3., 1.) 
              distance_weight_dict[stop_ind].append(weight_function)
          else:
              pass 
  return distance_weight_dict

def stop_density(stops_df,threshold = 2, reg = 0.01):
  """Computes the density of stops in a radius given by the threshold

  Args:
      stops_df (pd.DataFrame): Contains geographical locations of certain reference points (in particular, bus stops).
      threshold (int, optional):  Radius below which amenities are included in the final score. Defaults to 2.
      reg (float, optional): _description_. Regularization factor. Necessary when using weight function 1. Defaults to 0.01.

  Returns:
      _type_: _description_
  """
  stop_distances_dict = {}

  stop_long_col = [ele for ele in stops_df.columns if "lon" in ele][0]
  stop_lat_col = [ele for ele in stops_df.columns if "lat" in ele][0]

  for stop_ind_1 in stops_df.index:

      stop_1_long = stops_df.loc[stop_ind_1, stop_long_col]
      stop_1_lat = stops_df.loc[stop_ind_1, stop_lat_col]

      stop_distances_dict[stop_ind_1] = []

      for stop_ind_2 in stops_df[~stops_df.index.isin([stop_ind_1])].index:

        stop_2_long = stops_df.loc[stop_ind_2, stop_long_col]
        stop_2_lat = stops_df.loc[stop_ind_2, stop_lat_col]

        stops_dist = haversine(stop_1_long, stop_1_lat, stop_2_long, stop_2_lat)
        
        if reg < stops_dist < threshold:
            weight_function = 1/(reg + stops_dist)
            stop_distances_dict[stop_ind_1].append(weight_function)
        else:
            pass 
  return stop_distances_dict


def min_max_scaler(input_series):
    """Applies min-max scaler to an input series

    Args:
        input_series (pd.Series)

    Returns:
        np.array: Mix-max scaled version of the input series
    """
    scaler = MinMaxScaler()
    input_values = input_series.values.reshape(-1, 1)
    scaler.fit(input_values)
    output_values = scaler.transform(input_values).flatten()
    return output_values

    
def amenity_score_calc(lon1, lat1, amenities_df, amenities_relative_weigh_dict, threshold = 1.5, regularizer = 0.2):
    """Creates an amenity score for a given reference point.

    Args:
        lon1 (float): Longitude of the reference point
        lat1 (float): Latitude of the reference point
        amenities_df (pd.DataFrame): Contains all the amenities for reference
        amenities_relative_weigh_dict (dict): Contains all the amenities and their relative weights
        threshold (float, optional): Radius below which amenities are included in the final score. Defaults to 1.5.
        regularizer (float, optional): Regularization factor. Necessary when using weight function 1. Defaults to 0.2.

    Returns:
        float: Total amenity score the input reference point.
    """
    score_list = []
    
    for i in range(amenities_df.shape[0]):
        lat_amenity = amenities_df.loc[i, 'lat']
        lon_amenity = amenities_df.loc[i, 'lon']
        amenity_temp = amenities_df.loc[i, 'tags']

        if amenity_temp == "stop":
          score_multiplier = amenities_relative_weigh_dict[amenity_temp]*(1+amenities_df.loc[i, "Average Num Rides Normalized"])
        
        else:
          score_multiplier = amenities_relative_weigh_dict[amenity_temp]

        distance = haversine(lon_amenity, lat_amenity, lon1, lat1)
        #decay = normal_dist_decay(distance, mu = 1)

        if distance < threshold:
          score = score_multiplier/(regularizer + distance)**2
          score_list.append(score)
        else:
          pass

    return sum(score_list)

def get_region(point, df):
    """Gets population data for a certain point in a grid

    Args:
        point (geopd.point)
        df (geopd)

    Returns:
        float
    """
    return df[df['geometry'].apply(lambda x: x.contains(point)) == True]['N_INDIVIDUOS_RESIDENT'].to_list()[0]

def generating_stop_data(stop_times_df, trip_id = "trip_id_mod"):
    """Generates a time-series for a certain mean of transport

    Args:
        stop_times_df (pd.DataFrame): Contains arrival times of a certain mean of transport at a given stop
        trip_id (str, optional): Column identifier of the trip_id (bus/tram line - example, F train in New York). Defaults to "trip_id_mod".

    Returns:
        tuple: Tuple of dictionaries - the first contains the daily time series profile for rides on a given stop,
        the second contains the average number of rides per hour on a given stop.
    """
    full_stop_dict = {}
    stop_average_dict = {}

    porto_stops_list = stop_times_df["stop_id"].unique()

    for stop in porto_stops_list:
        var_df = stop_times_df[[trip_id,"stop_id","arrival_time"]][stop_times_df["stop_id"] == stop]
        var_df_2 = var_df.pivot_table(index = ["arrival_time","stop_id"], \
                        columns = [trip_id], values = [trip_id], aggfunc = len, fill_value = 0)\
                        .reset_index(level = "stop_id")
        var_df_3 = extract_stops_resample(var_df_2)
        full_stop_dict[stop] = var_df_3
        stop_average_dict[stop] = var_df_3.drop(columns = ["stop_id"]).sum(axis = 1).mean()
    
    return full_stop_dict , stop_average_dict

def computing_scooter_events(e_scooter_df):
    """Computing scooter ride times from scooter events.

    Args:
        e_scooter_df (pd.DataFrame): Contains e-scooter data.

    Returns:
        tuple: Tuple of dictionaries containing the: 1. start times of each ride, 2. end times of each ride,
        3. time duration of each ride, 4. length transverse on each ride.
    """
    average_scooter_speed = 20 ## Km/h

    time_duration_scooter_trips = {}
    length_duration_scooter_trips = {}
    length_mismatch = []

    start_times_dict = {}
    end_times_dict = {}

    for scooter_id in e_scooter_df["device_id"].unique():
        
        single_scooter_df = e_scooter_df[e_scooter_df["device_id"] == scooter_id].reset_index()

        trip_starts = (single_scooter_df["event_types"] == "{trip_start}")
        supposed_end_inds = [ele + 1 for ele in (trip_starts)[trip_starts].index]
        
        trip_ends = (single_scooter_df["event_types"] == "{trip_end}")
        actual_end_inds = [ele for ele in (trip_ends)[trip_ends].index if ele in supposed_end_inds]
        actual_start_inds = [ele - 1 for ele in actual_end_inds]
        
        relevant_inds = (actual_end_inds + actual_start_inds)
        relevant_inds.sort()

        relevant_df = single_scooter_df.loc[relevant_inds]
        relevant_start_df = relevant_df[relevant_df["event_types"] == "{trip_start}"].reset_index()
        relevant_end_df = relevant_df[relevant_df["event_types"] == "{trip_end}"].reset_index()
        
        start_times_dict[scooter_id] = relevant_start_df["timestamp"].dt.hour.values
        end_times_dict[scooter_id] = relevant_end_df["timestamp"].dt.hour.values

        time_duration_scooter_trips[scooter_id] = (relevant_end_df["timestamp"] - relevant_start_df["timestamp"])\
                .dt.seconds.values/(60*60) ## Ride time in hours

        length_duration_scooter_trips[scooter_id] = average_scooter_speed*time_duration_scooter_trips[scooter_id] ##Distance traveled in Km

    return start_times_dict, end_times_dict, time_duration_scooter_trips, length_duration_scooter_trips