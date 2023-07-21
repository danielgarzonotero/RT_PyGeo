#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Histogram 1 - RT
column_index = 1
df = pd.read_csv('dia.csv')

plt.hist(df.iloc[:, column_index], bins=100, color="lightcoral") 
plt.xlabel('Values RT (min)')
plt.ylabel('Frequency')
plt.title('Histogram RT - Rosenberger et al. Dataset')

# Calculate mean and standard deviation
data_mean = np.mean(df.iloc[:, column_index])
data_std = np.std(df.iloc[:, column_index])

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {data_mean:.2f} min\nStd: {data_std:.2f} min", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

""" df.to_csv('data/rt_dataset_reduced_10000.csv', index=False) """
# %%
