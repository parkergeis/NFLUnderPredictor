import numpy as np
import pandas as pd
from helper_functions import data_load, data_split, rf_model, google_export

year = 2024
week = 5
day = "Thursday"

allSeasons, currSeason = data_load(year, week)

features = allSeasons.drop(['Under', 'Push', 'gameday', 'game_id', 'surface', 'home_score', 'away_score', 'result', 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn', 'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee', 'stadium', 'wind', 'temp'], axis=1).columns

X_train, y_train, X_test, y_test = data_split(allSeasons, features, year, week, day)

prediction_df = rf_model(X_train, y_train, X_test)

# Predicted Plays log
nextPlays = pd.merge(right=prediction_df, left=currSeason, right_index=True, left_index=True, how='left')
nextPlays = nextPlays[nextPlays.Prediction == 1]
nextPlays = nextPlays[['game_id', 'season_x', 'week_x', 'home_team', 'away_team', 'gametime_x', 'weekday_x', 'total_line_x', 'under_odds_x']]
nextPlays.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Total Line', 'Under Odds']

google_export(nextPlays)