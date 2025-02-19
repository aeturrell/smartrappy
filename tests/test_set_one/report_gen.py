import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("output.xlsx")
plt.plot(df["x"], df["y"])
plt.savefig("plot.png")
