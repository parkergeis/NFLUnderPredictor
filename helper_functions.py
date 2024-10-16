import numpy as np
import pandas as pd
import nfl_data_py as nfl
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import gspread
from dicts import dict_nfl_teams, dict_day, dict_roof
import warnings
warnings.filterwarnings('ignore')

def data_load(year, week):
    # Variable declaration
    df = nfl.import_schedules(years=range(year-3,year+1))
    currSeason = df[df.season == year]

    # Prepare dataframe by dropping irrelevant predictors and formatting columns for KNN
    df['Under'] = np.where(df['total'] < df['total_line'], 1, 0)
    df['Push'] = np.where(df['total'] == df['total_line'], 1, 0)
    df = df[df.Push != 1]
    df.drop('Push', axis=1, inplace=True)

    # Function to convert time to seconds
    def time_to_seconds(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600 + minutes * 60
    df['gametime'] = df['gametime'].apply(time_to_seconds)

    df['home_team'].replace(dict_nfl_teams, inplace=True)
    df['away_team'].replace(dict_nfl_teams, inplace=True)
    df['weekday'].replace(dict_day, inplace=True)
    df['roof'].replace(dict_roof, inplace=True)

    return df, currSeason

def data_split(df, features, year, week, day="All"):
    train_df = df[(df.season < year) | ((df.season == year) & (df.week < week))]

    if day != "All":
        test_df = df[(df.season == year) & (df.week == week) & (df.weekday == day)]
    else:
        test_df = df[(df.season == year) & (df.week == week)]
    train_df.dropna(inplace=True)
    X_train = train_df[features]
    y_train = train_df.Under
    X_test = test_df[features]
    y_test = test_df.Under

    return X_train, y_train, X_test, y_test

def data_split_tune(df, features):
    df = df.dropna()
    y = df.Under
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    return X_train, y_train

def rf_model(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    X_test['Prediction'] = preds
    X_test['weekday'].replace(dict_day, inplace=True)

    return X_test

def xgb_tuning(X_train, y_train):
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 500, 1000],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.1, 0.5, 1.0]
    }

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_

def xgb_model(X_train, y_train, X_test, params):
    model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    **params
    )

    # Evaluation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict probabilities and classes on selected features
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Append predictions and probabilities to X_test
    X_test['Prediction'] = y_pred
    X_test['Probability'] = y_pred_proba

    return X_test

def google_export(df, title, sheet):
    gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
    sh = gc.open(title)

    # Add weekly plays
    worksheet1 = sh.worksheet(sheet)
    worksheet1.append_rows(df.values.tolist())