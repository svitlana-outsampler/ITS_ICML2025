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
#print(df_trend)

#print available columns
print(df_trend.columns)
# wait for the user to press enter
input("Press Enter to continue...")

# save it as json file
df_trend.to_json('qa_dataset_trend.json', orient='records')

import requests
# send a prompt question to the llm at http://localhost:8000/
# for a warming up
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


index = 0

max_index = df_trend.shape[0]
print("max_index: ", max_index, " index: ", index)

nb_success = 0

nb_questions = len(df_trend)

for index in range(nb_questions):
    print("index: ", index)


    preprompt = "Here is a time series : \n"
    # add the time series of the first question to the prompt
    prompt = preprompt + ', '.join(map(str, df_trend.iloc[index]['ts']))
    options = df_trend.iloc[index]['options']
    # convert the options to a string
    options = ', '.join(map(str, options))
    question = df_trend.iloc[index]['question']
    print("question: ", question, " options: ", options)
    prompt = prompt + "\n" + question + "\npossible answers: " + options + "\n"
    prompt = prompt + "Answer shortly the question by giving only one of the options.\n"

    expected_answer = df_trend.iloc[index]['answer']

    #print(prompt)
    #input("Press Enter to continue...")

    # define the new payload
    # limiting the number of tokens to 20
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20
    }
    response = requests.post(url, json=payload)
    # Print the response, which is extracted from the 'content' in the payload
    content = response.json()['choices'][0]['message']['content']
    stop_reason = response.json()['choices'][0]['finish_reason']
    if response.status_code == 200:
        print("LLM answer: ", content)
        #print(stop_reason)
    else:
        print(f"Error: {response.status_code}, {response.text}")

    print("Expected answer: ", expected_answer)
    #convert expected answer to string
    expected_answer = str(expected_answer)
    #convert content to string
    content = str(content)
    #convert the expected answer to lower case if the type permits
    expected_answer = expected_answer.lower()
    # convert the content to lower case if the type permits
    content = content.lower()
    # check if the expected answer is in the content
    if expected_answer in content:
        print("Correct")
        nb_success += 1
    else:
        print("Incorrect")

success_rate = nb_success / nb_questions
print("Success rate: ", success_rate)


