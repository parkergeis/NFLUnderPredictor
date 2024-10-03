import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_load import data_load

year = 2024
week = 5
day = "Thursday"

allSeasons, currSeason = data_load(year, week)

df = pd.get_dummies(allSeasons, drop_first=True, columns=['game_type', 'location', 'stadium_id', 'home_team', 'away_team'])

features = df.drop(['Under', 'Push', 'gameday', 'game_id', 'surface', 'home_score', 'away_score', 'result', 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn', 'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee', 'stadium', 'wind', 'temp'], axis=1).columns

train_df = df[(df.season < year) | ((df.season == year) & (df.week < week))]
test_df = df[(df.season == year) & (df.week == week)]
train_df.dropna(inplace=True)
X_train = train_df[features]
y_train = train_df.Under
X_test = test_df[features]
y_test = test_df.Under

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
X_test['Prediction'] = preds
# Predicted Plays log
nextPlays = pd.merge(right=X_test, left=currSeason, right_index=True, left_index=True, how='left')
nextPlays = nextPlays[nextPlays.Prediction == 1]
nextPlays = nextPlays[['game_id', 'season_x', 'week_x', 'home_team', 'away_team', 'gametime_x', 'weekday_x', 'total_line_x', 'under_odds_x']]
nextPlays.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Total Line', 'Under Odds']

# Value cleanup
dict_day = {"Day": {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}}
nextPlays.replace(dict_day, inplace=True)
nextPlays = nextPlays[nextPlays.Day == day]
# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("Over/Under NFL Model")

# Add weekly plays
worksheet1 = sh.worksheet("Plays")
worksheet1.append_rows(nextPlays.values.tolist())