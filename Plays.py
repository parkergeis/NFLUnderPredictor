import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import nfl_data_py as nfl
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
today = dt.date.today()
year = today.year

# Variable declaration
df = nfl.import_schedules(years=range(year-3,year+1))
currSeason = df[df.season == year]
predWeek = currSeason[['week', 'total']].dropna()
if np.isnan(predWeek.week.max()):
    predWeek = 1
else:
    predWeek = predWeek.week.max() + 1

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
dict_surface = {"surface": {"grass": 0, "grass ": 0, "fieldturf": 1, "astroturf": 2, "sportturf": 3, "matrixturf": 4, "astroplay": 5, "a_turf": 6, "dessograss": 7}}
df.replace(dict_surface, inplace=True)

df = pd.get_dummies(df, drop_first=True, columns=['game_type', 'location', 'stadium_id', 'home_team', 'away_team'])

features = df.drop(['Under', 'Push', 'gameday', 'game_id', 'home_score', 'away_score', 'result', 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn', 'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee', 'stadium', 'wind', 'temp'], axis=1).columns

train_df = df[(df.season < year) | ((df.season == year) & (df.week < predWeek))]
test_df = df[(df.season == year) & (df.week == predWeek)]
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

# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("Over/Under NFL Model")

# Add weekly plays
worksheet1 = sh.worksheet("Plays")
worksheet1.append_rows(nextPlays.values.tolist())