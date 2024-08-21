import pandas as pd
import nfl_data_py as nfl
import datetime as dt
today = dt.date.today()
year = today.year

# Variable declaration
df = nfl.import_schedules(years=[year])
# NFL Results log
finalScores = df[['game_id', 'total']].dropna()
finalScores.columns = ['Game ID', 'Total']

# Export to Sheets
import gspread
gc = gspread.service_account(filename='/Users/parkergeis/.config/gspread/seismic-bucksaw-427616-e6-5a5f28a2bafc.json')
sh = gc.open("Over/Under NFL Model")

# Add weekly results
worksheet2 = sh.worksheet("Results")
worksheet2.append_rows(finalScores.values.tolist())