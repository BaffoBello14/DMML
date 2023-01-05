import pandas as pd
df = pd.read_csv("vehicles.csv")
df = df.dropna(inplace=True)
print(df.info)
