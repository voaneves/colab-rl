import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot') # Customizando o estilo do matplotlib

fullpath = os.path.abspath(os.path.join('Training DQN.xlsx'))
training_data = pd.read_excel(fullpath)
rolling_data = training_data[['Mean size 10']].rolling(10)
data1 = rolling_data.mean()
data1.columns = ['Moving Average']
data2 = rolling_data.max()
data2.columns = ['Moving Max']
data2 = rolling_data.min()
data2.columns = ['Moving Min']
data3 = rolling_data.std()
data3.columns = ['Moving Std']

rolling_average_df = training_data.join([data1, data2, data3])

plt.plot(rolling_average_df[['Epoch']].values, rolling_average_df[['Moving Average']].values)
plt.fill_between(np.squeeze(rolling_average_df[['Epoch']].values),
                 np.squeeze(rolling_average_df[['Moving Average']].values-rolling_average_df[['Moving Std']].values),
                 np.squeeze(rolling_average_df[['Moving Average']].values+rolling_average_df[['Moving Std']].values), alpha=0.5)
plt.show()
