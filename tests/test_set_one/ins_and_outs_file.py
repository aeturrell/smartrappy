import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("lalala.csv")

with open("text.txt", "w") as f:
    f.write("blah")

df.to_csv("out.csv")

fig, ax = plt.subplots()
ax.plot([1, 2, 4], [3, 4, 5])
plt.savefig("out_figure.svg")
