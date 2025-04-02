# basic class to handle the exam time series dataset
import pandas as pd
import requests


def check_answer(expected_answer, content):
    # check if the expected answer is in the content
    # convert expected answer to string
    expected_answer = str(expected_answer)
    # convert content to string
    content = str(content)
    # check if the expected answer is in the content
    if expected_answer in content:
        return True
    else:
        return False

class LLMExam:
    def __init__(self):
        with open('qa_dataset.json', 'r') as f:
            data = f.read()
        df = pd.read_json(data)
        # select all entries with "subcategory": "Trend Recognition"
        self.df_trend = df[df['subcategory'] == 'Trend Recognition']
        self.df_trend.to_json('qa_dataset_trend.json', orient='records')

        import requests
        # send a prompt question to the llm at http://localhost:8000/
        # for a warming up
        self.url = "http://127.0.0.1:8080/v1/chat/completions"
        # Define the payload
        payload = {
            "model": "llama",  # Adjust based on your model's name
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "temperature": 0.7
        }

        response = requests.post(self.url, json=payload)

        # Print the response
        if response.status_code == 200:
            print(response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")

    def get_question(self, index):
        preprompt = "Here is a time series : \n"
        # add the time series of the first question to the prompt
        prompt = preprompt + ', '.join(map(str, self.df_trend.iloc[index]['ts']))
        options = self.df_trend.iloc[index]['options']
        # convert the options to a string
        options = ', '.join(map(str, options))
        question = self.df_trend.iloc[index]['question']
        #print("question: ", question, " options: ", options)
        prompt = prompt + "\n" + question
        prompt = prompt + "\nAnswer shortly the question by giving only one of the following:\n"
        prompt = prompt + options + "\n"
        expected_answer = self.df_trend.iloc[index]['answer']
        return prompt, expected_answer

    def get_llm_answer(self, prompt):
        # Define the payload
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20
        }
        response = requests.post(self.url, json=payload)
        # Print the response, which is extracted from the 'content' in the payload
        content = response.json()['choices'][0]['message']['content']
        stop_reason = response.json()['choices'][0]['finish_reason']
        if response.status_code == 200:
            pass
            #print("LLM answer: ", content)
            #print(stop_reason)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            print("stop reason: ",stop_reason)
        return content
    
    def benchmark(self):
        print("-------------------------------------------------")
        print("Benchmark start ---------------------------------")
        nb_questions = len(self.df_trend)
        nb_success = 0
        for index in range(nb_questions):
            prompt, expected_answer = self.get_question(index)
            llm_answer = self.get_llm_answer(prompt)
            if check_answer(llm_answer, expected_answer):
                nb_success += 1
            print("index: ", index, " nb_success: ", nb_success)
            # print the tronc first and last characters of the prompt
            # separated by "..."
            tronc = 50
            print("prompt: ", prompt[:tronc], "...", prompt[-2*tronc:])
            print("llm_answer: ", llm_answer)
            print("expected_answer: ", expected_answer)
            print("nb_success: ", nb_success)
            print("nb_questions: ", nb_questions)
            print("success rate: ", nb_success / (index+1),"--------------------------")
        print("Benchmark end ---------------------------------")
        print("-----------------------------------------------")
        return nb_success / nb_questions
        

bench = LLMExam()
# index = 12
# prompt, expected_answer = bench.get_question(index)
# print(prompt)
# print("Expected answer: ", expected_answer)

# llm_answer = bench.get_llm_answer(prompt)
# print("LLM answer: ",llm_answer)

success_rate = bench.benchmark()

print("success rate: ", success_rate)