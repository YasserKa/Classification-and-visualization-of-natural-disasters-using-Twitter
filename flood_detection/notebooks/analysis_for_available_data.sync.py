# %%
import matplotlib.pyplot as plt
import pandas as pd
import json

# %%
with open("output/annotated_tweets_extracted.json", "r") as file:
    tweets_json = json.load(file)
df = pd.json_normalize(list(tweets_json.values()))
print(df.shape)


# %%
def value_counts_lambda(col):
    return print(df[col].value_counts())


value_counts_lambda("Informative/relevant/non sarcastic")
value_counts_lambda("Contains specific information about IMPACTS")

# %%
df = df[(df['On Topic'] != '') & (df['Informative/relevant/non sarcastic']
                                  != '') & (df['Contains specific information about IMPACTS'] != '')]
df['created_at'] = pd.to_datetime(df['created_at'])

df['On Topic'] = df['On Topic'].astype(int)
df['Informative/relevant/non sarcastic'] = df['Informative/relevant/non sarcastic'].astype(
    int)
df['Contains specific information about IMPACTS'] = df['Contains specific information about IMPACTS'].astype(
    int)

df_gb = df.groupby([df["created_at"].dt.date])

# %%
fig, ax1 = plt.subplots()
dates = df_gb.groups.keys()

plt.plot(dates, df_gb['On Topic'].sum().values)
plt.plot(dates, df_gb["Informative/relevant/non sarcastic"].sum().values)
plt.plot(
    dates, df_gb["Contains specific information about IMPACTS"].sum().values)

plt.title('Time series for tweets')
plt.xlabel('Date')
plt.ylabel('count')

plt.show()
