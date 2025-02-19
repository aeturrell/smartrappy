import pandas as pd

df = pd.read_csv("data.csv")
df.to_excel("output.xlsx")
