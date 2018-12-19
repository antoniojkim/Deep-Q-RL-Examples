import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./Cart-Pole/rewards_log.csv")

column = "cumulative_reward"
print(df[column].max())
print(df[column].idxmax())

plt.plot(df["episode"], df[column], c=f"C0")
plt.plot(df["episode"], np.polyval(np.polyfit(df["episode"], df[column], 1), df["episode"]), c=f"C1", label=column)

plt.show()

# df = pd.read_csv("./Cart-Pole/loss_log.csv")
# print(df["loss"].max())
# print(df["loss"].idxmax())

# plt.plot(df["iteration"], df[f"loss"], c=f"C0")
# plt.plot(df["iteration"], np.polyval(np.polyfit(df["iteration"], df[f"loss"], 1), df["iteration"]), c=f"C1", label=f"loss")

# plt.show()