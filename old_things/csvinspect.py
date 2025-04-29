# I need to open a large csv file in python

import pandas as pd
df = pd.read_csv('data.csv')

# print number of rows and columns
print(df.shape)

# save the first 100 rows to a new csv file
df.head(100).to_csv('data_100.csv', index=False)
# save the last 100 rows to a new csv file
df.tail(100).to_csv('data_last_100.csv', index=False)
# merge the two new csv files
df_merged = pd.concat([pd.read_csv('data_100.csv'), pd.read_csv('data_last_100.csv')])
# save the merged csv file
df_merged.to_csv('data_merged.csv', index=False)

# plot the fourth column of data. Take the first 100 rows and plot the fourth column take index as x variable

import matplotlib.pyplot as plt
plt.plot(df_merged.iloc[:100, 3])
plt.show()