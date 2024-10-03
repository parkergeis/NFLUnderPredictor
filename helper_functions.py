import numpy as np
import pandas as pd
import nfl_data_py as nfl

def data_load(year, week):
    # Variable declaration
    df = nfl.import_schedules(years=range(year-3,year+1))
    currSeason = df[df.season == year]

    # Prepare dataframe by dropping irrelevant predictors and formatting columns for KNN
    df['Under'] = np.where(df['total'] < df['total_line'], 1, 0)
    df['Push'] = np.where(df['total'] == df['total_line'], 1, 0)
    df = df[df.Push != 1]

    def date_to_month(time_str):
        year, month, day = map(int, time_str.split('-'))
        return month
    df['month'] = df['gameday'].apply(date_to_month)

    # Function to convert time to seconds
    def time_to_seconds(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600 + minutes * 60
    
    # Apply the function to the 'time' column
    df['gametime'] = df['gametime'].apply(time_to_seconds)

    dict_day = {"weekday": {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}}
    df.replace(dict_day, inplace=True)
    dict_roof = {"roof": {"outdoors": 0, "dome": 1, "closed": 2, "open": 3}}
    df.replace(dict_roof, inplace=True)

    df = pd.get_dummies(df, drop_first=True, columns=['game_type', 'location', 'stadium_id', 'home_team', 'away_team'])

    return df, currSeason

def data_split(df, features, year, week):
    train_df = df[(df.season < year) | ((df.season == year) & (df.week < week))]
    test_df = df[(df.season == year) & (df.week == week)]
    train_df.dropna(inplace=True)
    X_train = train_df[features]
    y_train = train_df.Under
    X_test = test_df[features]
    y_test = test_df.Under

    return X_train, y_train, X_test, y_test