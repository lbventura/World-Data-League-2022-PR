import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def creating_waste_features(waste_df):
    """Creating time-series features

    Args:
        waste_df (pd.DataFrame): Contains waste information

    Returns:
        pd.DataFrame
    """
    waste_ext_df = waste_df.sort_values('report_date').reset_index().drop("index", axis = 1).copy()

    waste_ext_df['load_weight'].fillna(np.nan, inplace  = True)
    waste_ext_df['load_weight'] = waste_ext_df['load_weight'].str.replace(',','').astype("float64")

    waste_ext_df["Report Date Year"] = waste_ext_df['report_date'].dt.year
    waste_ext_df["Report Date Month"] = waste_ext_df['report_date'].dt.month
    waste_ext_df["Report Date Day"] = waste_ext_df['report_date'].dt.day
    waste_ext_df["Report Date Weekday"] = waste_ext_df['report_date'].dt.weekday
    waste_ext_df['Report Date Week'] = waste_ext_df['report_date'].dt.isocalendar().week

    return waste_ext_df


def generating_waste_charts(waste_df, start_year = 2012, end_year = 2022):
    """Generates yearly time-series charts

    Args:
        waste_df (pd.DataFrame): Contains waste data
        start_year (int, optional): Defaults to 2012.
        end_year (int, optional): Defaults to 2022.

    Returns:
        tuple: Tuple of dictionaries. The first contains the total amount of waste
        as values (with years as keys). The second contains a pd.Series object with
        types of waste and disposal location as index, column the amount of waste.
        The keys of the dictionary are years.
    """

    year_dict = {}
    total_trash_year_cat_dict = {}
    total_trash_year_dict = {}

    for choice_year in range(start_year,end_year):

        df_y = waste_df.loc[waste_df['report_date'].dt.year == choice_year,:].reset_index().drop("index", axis = 1)

        df_y["Report Date Month"] = df_y['report_date'].dt.month
        
        total_trash_df_cat_y = df_y.groupby(['load_type','dropoff_site']).agg({'load_weight':"sum"})
        
        total_trash_year_cat_dict[choice_year] = total_trash_df_cat_y
        
        total_trash_df_y = total_trash_df_cat_y.groupby('load_type').sum().T 
        
        total_trash_df_y.index = [choice_year]
        
        total_trash_year_dict[choice_year] = total_trash_df_y
        
        df_garb_y = df_y.groupby('report_date').agg({'load_weight': "sum"})
        
        df_garb_y.plot()
        plt.show()
    
    return total_trash_year_dict , total_trash_year_cat_dict