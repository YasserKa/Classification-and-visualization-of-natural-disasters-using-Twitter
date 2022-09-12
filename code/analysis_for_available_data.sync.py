# %%
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import json

# %%
with open(f"output/annotated_tweets_extracted.json", "r") as file:
    tweets_json = json.load(file)

# %%
df = pd.json_normalize(tweets_json.values())
df = df[(df['On Topic'] != '') & (df['Informative/relevant/non sarcastic'] != '') & (df['Contains specific information about IMPACTS'] != '')]
df['created_at'] =  pd.to_datetime(df['created_at'])

df['On Topic'] =  df['On Topic'].astype(int)
df['Informative/relevant/non sarcastic'] =  df['Informative/relevant/non sarcastic'].astype(int)
df['Contains specific information about IMPACTS'] =  df['Contains specific information about IMPACTS'].astype(int)

df_gb = df.groupby([df["created_at"].dt.date])

# %%
pd.set_option('display.max_colwidth', 100)
df[df['created_at'].dt.date == datetime.datetime(2021,7, 16).date()]['tweet_url']

# %%
df_gb['On Topic'].sum().sort_values()

# %%
fig, ax1 = plt.subplots()
x = df_gb['On Topic'].sum().keys()

ax1.set_xlabel('date')
ax1.set_ylabel('Y1-axis', color = 'red') 
ax1.plot(x, df_gb['On Topic'].sum().values, color = 'red') 
ax1.tick_params(axis ='y', labelcolor = 'red') 

ax2 = ax1.twinx() 

ax2.set_ylabel('Y2-axis', color = 'blue') 
ax2.plot(x, df_gb["Informative/relevant/non sarcastic"].sum().values, color = 'blue') 
ax2.tick_params(axis ='y', labelcolor = 'blue') 

ax3 = ax1.twinx() 

ax3.set_ylabel('Y3-axis', color = 'purple') 
ax3.plot(x, df_gb["Contains specific information about IMPACTS"].sum().values, color = 'purple') 
ax3.tick_params(axis ='y', labelcolor = 'purple') 

plt.show()
