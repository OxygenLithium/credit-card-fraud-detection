# Exploratory data analysis for visualizing input data
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("src/data/creditcard.csv")

print(f"Shape: {df.shape}")
print("Info:")
print(df.info())
print("Head:")
print(df.head())

non_fraud = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]
print(f"Non-fraud cases: {len(non_fraud)}")
print(f"Fraud cases: {len(fraud)}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(non_fraud["Time"], bins=30, alpha=0.7, color='blue', label='Non-fraud')
axes[0, 0].set_title("Time Distribution - Non-fraud Cases")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(non_fraud["Amount"], bins=30, alpha=0.7, color='blue', label='Non-fraud')
axes[0, 1].set_title("Amount Distribution - Non-fraud Cases")
axes[0, 1].set_xlabel("Amount")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_yscale("log")

axes[1, 0].hist(fraud["Time"], bins=30, alpha=0.7, color='red', label='Fraud')
axes[1, 0].set_title("Time Distribution - Fraud Cases")
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(fraud["Amount"], bins=30, alpha=0.7, color='red', label='Fraud')
axes[1, 1].set_title("Amount Distribution - Fraud Cases")
axes[1, 1].set_xlabel("Amount")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_yscale("log")

plt.tight_layout()
plt.show()
