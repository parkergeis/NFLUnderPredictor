import pandas as pd
import nfl_data_py as nfl
import datetime as dt
today = dt.date.today()
year = today.year

# Variable declaration
df = nfl.import_schedules(years=range(year-5,year+1))
# NFL Results log
finalScores = df[['game_id', 'season', 'week', 'home_team', 'home_score', 'away_team', 'away_score', 'total_line', 'total', 'over_odds', 'under_odds', 'spread_line', 'result']].dropna()

# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("Over/Under NFL Model")

# Add weekly results
worksheet2 = sh.worksheet("Results")
total_rows = worksheet2.row_count
if total_rows > 1:
    worksheet2.batch_clear([f"A2:M{total_rows}"])
worksheet2.append_rows(finalScores.values.tolist())