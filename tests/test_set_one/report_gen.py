import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel("output.xlsx")
plt.plot(df['x'], df['y'])
plt.savefig("plot.png")