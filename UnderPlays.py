import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
df = df[['home_team', 'away_team', 'season', 'total', 'game_type', 'week', 'gameday', 'gametime', 'location', 'away_moneyline', 'home_moneyline', 'spread_line', 'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds', 'div_game', 'roof', 'surface', 'referee', 'stadium_id']]
df['Over'] = np.where(df['total'] > df['total_line'], 1, 0)
df['Under'] = np.where(df['total'] < df['total_line'], 1, 0)
df['Push'] = np.where(df['total'] == df['total_line'], 1, 0)
df = df[df.Push != 1]
df.drop(columns='total', inplace=True)
df['gameday'] = pd.to_datetime(df['gameday'])
df['DayOfWeek'] = df['gameday'].dt.day_of_week
# Function to convert time to seconds
def time_to_seconds(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60

# Apply the function to the 'time' column
df['gametime'] = df['gametime'].apply(time_to_seconds)
df = pd.get_dummies(df, drop_first=True,columns=['game_type', 'location', 'roof', 'surface', 'referee', 'stadium_id'])
df.reset_index(drop=True, inplace=True)
df = df.dropna()

# Model building
feats = df.drop(columns=['home_team', 'away_team', 'season', 'gameday', 'Over', 'Under', 'Push'])
features = feats.columns
target = 'Under'

train_df = df[(df.season < year) & (df.week < predWeek) | (df.season < year)]
test_df = df[(df.season == year) & (df.week == predWeek)]
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

model = KNeighborsClassifier(n_neighbors=13)
classif = model.fit(X_train, y_train)
y_pred = classif.predict(X_test)

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