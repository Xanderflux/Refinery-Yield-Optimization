# Getting dataset ready

import pandas as pd
from scipy.io import arff

data = arff.loadarff('data.arff')
df= pd.DataFrame(data[0])
# print(df.head())

# 1. Rename columns to Engineering Terms
# Based on the metadata we found earlier
df = df.rename(columns={
    'u1': 'Top_Temp',          # Top Column Temperature
    'u2': 'Top_Pressure',      # Top Column Pressure
    'u3': 'Reflux_Flow',       # Reflux Flow Rate
    'u4': 'Flow_Next',         # Flow to Next Process
    'u5': 'Tray_6_Temp',       # 6th Tray Temperature
    'u6': 'Bottom_Temp',       # Bottom Temperature
    'u7': 'Bottom_Temp_Red',   # Redundant Bottom Temp
    'y':  'Butane_Yield'       # TARGET: Butane content (Lower is usually better/purer)
})

# 2. Inspect the new DataFrame
print("✅ Columns Renamed Successfully")
print(df.head())

# 3. Check for Missing Values (Crucial for cleaning)
print("\nMissing Values Count:")
print(df.isnull().sum())

# 4. Get Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# 5. Export to Readable data form (csv and excel)
df.to_excel('data.xlsx', index=False)
df.to_csv('data.csv', index=False)
