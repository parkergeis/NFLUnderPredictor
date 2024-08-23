import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import nfl_data_py as nfl
import datetime as dt
today = dt.date.today()
year = today.year

# Variable declaration
df = nfl.import_schedules(years=range(2000,year+1))
currSeason = df[df.season == year]
predWeek = currSeason[['week', 'total_line']].dropna()
predWeek = predWeek.week.max()

# Dataframe prep for modeling
df = df[['home_team', 'away_team', 'season', 'total', 'week', 'gametime', 'spread_line', 'total_line', 'under_odds']]
df['Over'] = np.where(df['total'] > df['total_line'], 1, 0)
df['Under'] = np.where(df['total'] < df['total_line'], 1, 0)
df['Push'] = np.where(df['total'] == df['total_line'], 1, 0)
df = df[df.Push != 1]
df.drop(columns='total', inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.dropna()

# Model building
features = ['spread_line', 'total_line', 'under_odds']
target = 'Under'

train_df = df[(df.season < year) & (df.week < predWeek) | (df.season < year)]
test_df = df[(df.season == year) & (df.week == predWeek)]
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=7))
])

classif = pipe.fit(X_train, y_train)

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('lof', LocalOutlierFactor(novelty=True))
])

pipe2.fit(X_train)
y_test_nov = pipe2.predict(X_test)

mask = [y == 1 for y in y_test_nov]

X_test = X_test[mask]
y_test = y_test[mask]
y_pred = classif.predict(X_test)
y_true = y_test

# Predicted Plays log
nextPlays = currSeason[currSeason.week == predWeek]
nextPlays['Predicted Outcome'] = y_pred
nextPlays = nextPlays[nextPlays['Predicted Outcome'] == 1]
nextPlays = nextPlays[['game_id', 'season', 'week', 'home_team', 'away_team', 'gametime', 'weekday', 'total_line', 'under_odds']]
nextPlays.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Total Line', 'Under Odds']

# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("Over/Under NFL Model")

# Add weekly plays
worksheet1 = sh.worksheet("Plays")
worksheet1.append_rows(nextPlays.values.tolist())