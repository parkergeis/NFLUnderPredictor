import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
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
    predWeek = predWeek.week.max()


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
dict_surface = {"surface": {"grass": 0, "grass ": 0, "fieldturf": 1, "astroturf": 2, "sportturf": 3, "matrixturf": 4, "astroplay": 5, "a_turf": 6, "dessograss": 7}}
df.replace(dict_surface, inplace=True)

df = pd.get_dummies(df, drop_first=True, columns=['game_type', 'location', 'stadium_id'])
df.reset_index(drop=True, inplace=True)

# Feature selection
feat_df = df.dropna()
X_variables = feat_df.drop(['Under', 'game_id', 'gameday', 'away_team', 'away_score', 'home_team', 'home_score', 'result', 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn', 'temp', 'wind', 'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee', 'stadium'], axis=1).copy()
y_variable = feat_df['Under'].copy()

selected_X = SelectKBest(f_classif, k=16)
selected_X.fit(X_variables, y_variable)

indices = selected_X.get_support(indices=True)
selected_features = X_variables.columns[indices]

# # Model building
train_df = df[(df.season < year) & (df.week < predWeek) | (df.season < year)]
test_df = df[(df.season == year) & (df.week == predWeek)]
X_train = train_df[selected_features]
y_train = train_df.Under
X_test = test_df[selected_features]
y_test = test_df.Under

# Define pipeline for use after split
pipe = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(pipe)
parameters = {'knn__n_neighbors': np.arange(1,50)}

knncv = GridSearchCV(estimator=pipeline,
                     param_grid=parameters,
                     n_jobs=-1,
                     cv=5)

classif = knncv.fit(X_train,y_train)

# pipe2 = Pipeline([
#     ('scaler', StandardScaler()),
#     ('lof', LocalOutlierFactor(novelty=True))
# ])

# pipe2.fit(X_train)
# y_test_nov = pipe2.predict(X_test)

# mask = [y == 1 for y in y_test_nov]

# X_test = X_test[mask]
# y_test = y_test[mask]
y_pred = classif.predict(X_test)
y_true = y_test
X_test['Prediction'] = y_pred

# Predicted Plays log
nextPlays = pd.merge(right=X_test, left=test_df, right_index=True, left_index=True, how='left')
nextPlays = nextPlays[nextPlays.Prediction == 1]
nextPlays = nextPlays[['game_id', 'season', 'week_x', 'home_team', 'away_team', 'gametime', 'weekday', 'total_line', 'under_odds_x']]
nextPlays.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Total Line', 'Under Odds']

# Value cleanup
dict_day = {"Day": {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}}
nextPlays.replace(dict_day, inplace=True)
def seconds_to_hhmm(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 12:
        hours = hours - 12
    return f"{hours}:{minutes:02}"
nextPlays['Start Time'] = nextPlays['Start Time'].apply(seconds_to_hhmm)

# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("Over/Under NFL Model")

# Add weekly plays
worksheet1 = sh.worksheet("Plays")
worksheet1.append_rows(nextPlays.values.tolist())