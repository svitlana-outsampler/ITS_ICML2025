import pandas as pd

# open file qa_dataset.json
with open('qa_dataset.json', 'r') as f:
    data = f.read()
# print data

# convert data to pandas dataframe
df = pd.read_json(data)

# select all entries with "subcategory": "Trend Recognition"
df_trend = df[df['subcategory'] == 'Trend Recognition']
# print df_trend
print(df_trend)

# save it as json file
df_trend.to_json('qa_dataset_trend.json', orient='records')

import requests
# send a prompt question to the llm at http://localhost:8000/
url = "http://127.0.0.1:8080/v1/chat/completions"
# Define the payload
payload = {
    "model": "llama",  # Adjust based on your model's name
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.7
}

response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
preprompt = "Here is a time series : \n"
# add the time series of the first question to the prompt
prompt = preprompt + ', '.join(map(str, df_trend.iloc[0]['ts']))
prompt = prompt + "what is the general trend (increasing or decreasing) of the time series?\n"

print(prompt)

# define the new payload

payload = {
    "model": "llama",  # Adjust based on your model's name
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7
}
response = requests.post(url, json=payload)
# Print the response
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")


