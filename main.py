import numpy as np
import pandas as pd
from helper_functions import data_load, data_split, data_split_tune, rf_model, xgb_tuning, xgb_model, google_export

year = 2024
week = 1
#day = 0 # 0-Sun, 1-Mon, 4-Thu

allSeasons, currSeason = data_load(year, week)

features = ['season', 'week', 'weekday', 'gametime', 'away_team', 'home_team', 'away_rest', 'home_rest', 'away_moneyline', 'home_moneyline', 'spread_line', 'total_line', 'under_odds', 'over_odds', 'div_game']

X_train, y_train, X_test, y_test = data_split(allSeasons, features, year, week)

prediction_df_rf = rf_model(X_train, y_train, X_test)

X_tune, y_tune = data_split_tune(allSeasons, features)
params = xgb_tuning(X_train, y_train)
prediction_df_xgb = xgb_model(X_train, y_train, X_test, params)

# Random Forest Plays log
# nextPlays_rf = pd.merge(right=prediction_df_rf, left=currSeason, right_index=True, left_index=True, how='left')
# nextPlays_rf = nextPlays_rf[nextPlays_rf.Prediction == 1]
# nextPlays_rf = nextPlays_rf[['game_id', 'season_x', 'week_x', 'home_team_x', 'away_team_x', 'gametime_x', 'weekday_x', 'total_line_x', 'under_odds_x']]
# nextPlays_rf.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Total Line', 'Under Odds']
# google_export(nextPlays_rf, "Over/Under NFL Model", "Plays")

# X Gradient Boosted Plays log
nextPlays_xgb = pd.merge(right=prediction_df_xgb, left=currSeason, right_index=True, left_index=True, how='left')
nextPlays_xgb = nextPlays_xgb[nextPlays_xgb.Prediction == 1]
nextPlays_xgb = nextPlays_xgb[['game_id', 'season_x', 'week_x', 'home_team_x', 'away_team_x', 'gametime_x', 'weekday_x', 'total_line_x', 'under_odds_x', 'Probability']]
nextPlays_xgb.columns = ['Game ID', 'Season', 'Week', 'Home', 'Away', 'Start Time', 'Day', 'Total Line', 'Under Odds', 'Probability']
google_export(nextPlays_xgb, "Over/Under NFL Model", "XGB Plays")