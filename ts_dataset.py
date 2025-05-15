import base64
import requests
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter



np.random.seed(42)

prompt_ts = r"""
Describe the time series in three sentences. First sentence: describe trend (increasing/decreasing/flat). Second sentence: noise intensity (low/medium/high). Third sentence: approximate localisation of global maximum (beginning/middle/end) and global minimum (beginning/middle/end).
Put the description in a JSON format with the following pattern
```json
{ "trend": <sentence1>,
  "noise": <sentence2>,
  "extrema": <sentence3> }
```
"""


# check that a json object can be extracted from a string and that
# the json object has the keys "trend", "noise", and "extrema"
# additional keys are allowed
def check_json_format(json_string):
    try:
        # Attempt to strip potential markdown code block fences ```json ... ``` or ``` ... ```
        if json_string.strip().startswith("```json"):
            json_string = json_string.strip()[7:-3].strip()
        elif json_string.strip().startswith("```"):
             json_string = json_string.strip()[3:-3].strip()
        
        print("json_string=", json_string)
        json_object = json.loads(json_string)
        if "trend" in json_object and "noise" in json_object and "extrema" in json_object:
            return True
        else:
            print("Warning: JSON object missing required keys.")
            return False
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to decode JSON: {e}")
        return False
    except Exception as e: # Catch other potential errors during processing
        print(f"Warning: An error occurred during JSON check: {e}")
        return False

# tools for detecting semantic similarity and contradictions:
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Load sentence transformer model
model_name = "paraphrase-MiniLM-L6-v2"
print("Loading model " + model_name)
model_st = SentenceTransformer(model_name)

# Load a model trained for contradiction detection (NLI)
nli_model_name = "roberta-large-mnli"
print("Loading NLI model " + nli_model_name)
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def compute_semantic_similarity(gold_output, generated_output):
    emb_generated = model_st.encode([generated_output], convert_to_tensor=True)
    emb_gold = model_st.encode([gold_output], convert_to_tensor=True)

    cosine_score = torch.nn.functional.cosine_similarity(emb_generated, emb_gold).cpu().item()
    euclidean_score = euclidean_distances(emb_generated.cpu(), emb_gold.cpu())[0][0]

    return cosine_score, euclidean_score

def detect_contradiction_nli(premise, hypothesis):
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    #labels = ['entailment', 'neutral', 'contradiction']
    labels = ['no', 'bof', 'ok'] 
    max_idx = torch.argmax(probs).item()
    return labels[max_idx], probs[0][max_idx].item()

# a small class for generating Orstein Uhlenbeck process together with an automatic
# human readible description of the process
class OUProcess:
    def __init__(self):
        self.seed = 0
        # give the seed to numpy
        self.T=1000          # Time horizon
        self.dt=0.1         # Time step
        self.theta=0.7      # Speed of mean reversion: higher values pull X(t) faster towards the mean
        self.slope = 0.0    # Slope of the mean curve
        self.mu=0.0         # Long-term mean: the value to which the process tends to revert
        self.sigma=0.      # Volatility: the intensity of the random fluctuations (Brownian motion component)
        self.x0=0.0         # Initial value of the process
        self.T=1.0          # Total time

    def generate(self, imagename="ou_process.png", filename="ou_process.dat"):
        #print(self.dt)
        N = 128
        self.dt = self.T/N
        #print(self.dt)
        # N = int(self.T / self.dt) # This line is redundant as N is already defined
        t = np.linspace(0, self.T, N)      # Time grid
        ti = np.arange(0, N)  # Time grid for plotting
        X = np.zeros(N)               # Array to store the process values

        # list of three sentence fact checked sentences
        sentences = []
        description = dict()
        # Apply random variations to parameters for diversity
        # Ensure parameters stay within reasonable bounds if necessary
        theta = max(0.1, self.theta + np.random.uniform(-0.5, 0.5)) # Keep theta positive
        mu = self.mu + np.random.uniform(-1,1)*2
        mu = 0.5
        sigma = self.sigma + np.random.uniform(0., 0.4)
        slope = self.slope+ np.random.uniform(-1,1)*0.25
        #sigma = 0 # Keep this commented unless you specifically want no noise
        #x0 = self.x0 + np.random.uniform(-1,1)*2
        x0 = 0.5
        X[0] = x0                     # Initial value

        if slope > sigma*0.1:
            description['trend'] = "the time series shows an overall increasing trend."
        elif slope < -sigma*0.1:
            description['trend'] = "the time series shows an overall decreasing trend."
        else:
            description['trend'] = "the time series shows no clear trend."


        if sigma < 0.1:
            description['noise']="the time series presents many small fluctuations."
        elif sigma > 0.3:
            description['noise']="the time series presents many large fluctuations."
        else:
            description['noise']="the time series presents many moderate fluctuations."


        # Use a fixed seed for reproducibility *within a single generation* if needed
        # np.random.seed(self.seed) # Uncomment if you want deterministic generation based on seed
        
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(self.dt))  # Brownian increment
            #X[i] = X[i-1] + theta * (mu - X[i-1]) * self.dt + sigma * dW
            X[i] = X[i-1] + slope * self.dt + sigma * dW
            # remet X dans l'intervalle [0,1]
            X[i] = np.clip(X[i], 0, 1)
        
        # get the max of the sequence
        #max_value = np.max(X)
        # get the min of the sequence
        #min_value = np.min(X)
        # scale the sequence so that all the values are between 0 and 99.9999
        X = X * 99.9999
        # round the values to 0 decimal places and convert to integer
        X = np.floor(X).astype(int)

                # recherche de la localisation en t du maximum et du minimum
        pos_max = np.argmax(X)
        print('pos_max', pos_max)
        if pos_max < 32:
            pos_desc = "The maximum is reached around the beginning part of the time series"
        elif pos_max > 96:
            pos_desc = "The maximum is reached towards the end of the time series"
        else:
            pos_desc = "The maximum is reached around the middle of the time series"

        pos_min = np.argmin(X)
        print('pos_min', pos_min)
        if pos_min < 32:
            pos_desc += " and the minimum is reached around the beginning part of the time series."
        elif pos_min > 96:
            pos_desc += " and the minimum is reached towards the end of the time series."
        else:
            pos_desc += " and the minimum is reached around the middle of the time series."

        description['extrema'] = pos_desc
        description['parameters'] = "sigma="+str(sigma) + " mu=" + str(mu) + " x0=" + str(x0) + " slope=" + str(slope)

        # convert to a json string
        description = json.dumps(description)
        
        # Save the process to a file
        np.savetxt(filename, X)
        
        plt.figure(figsize=(10, 5))
        # fix the y scale to between 0 and 100
        plt.ylim(0, 100)
        # fix the aspect ration to 1
        plt.gca().set_aspect('equal', adjustable='box')
        # plot the time series with spline interpolation

        #X = savgol_filter(X, window_length=1, polyorder=1)

        plt.plot(ti, X )
        plt.title('Time Series')
        plt.xlabel('Time')
        plt.ylabel('X(t)')
        plt.grid(True)
        # now save the picture to a png file
        plt.savefig(imagename)
        plt.close() # Close the plot to free memory
        return X, description

# définir les modèles: pixtral, mistral, qwen, nollm (génération de phrases factuelles)
# appeler ask ou ask noimage en fonction du modèle
# fact checker 

# a simple class for asking questions to a LLM about a given picture
# the picture is encoded in base64 and sent to the LLM
class Mistral:
    def __init__(self, dryrun = False, image = False):
        self.dryrun = dryrun
        self.image = image
        if image:
            self.api_key = os.environ.get("MISTRAL_API_KEY")
            if not self.api_key and not dryrun:
                raise ValueError("MISTRAL_API_KEY environment variable not set.")
            self.api_url = "https://api.mistral.ai/v1/chat/completions"
            #self.model = "mistral-large-latest" # Using the recommended model for function calling / JSON mode
            #self.model = "pixtral-large-latest"
            self.model = "pixtral-12b-2409"
            #self.model = "pixtral-large-latest" # Use pixtral if image input is definitely needed later
            print("Mistral initialized")
        else:
            self.api_key = os.environ.get("TEXTSYNTH_API_KEY")
            if not self.api_key and not dryrun:
                raise ValueError("TEXTSYNTH_API_KEY environment variable not set.")
            self.api_url = "https://palgania.ovh:8106/v1/chat/completions"
            self.model = "qwen3-30b-a3b" # Using the recommended model for function calling / JSON mode
            print("Palgania initialized")

        #print(f"Mistral API key: {'Set' if self.api_key else 'Not Set'}")
        print(f"LLM API URL: {self.api_url}")
        print(f"LLM model: {self.model}")
        print(f"Dry run: {self.dryrun}")
    
    def encode_image(self, image_path):
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
        
    def ask_noimage(self, question):
        """Ask a question using the model without an image."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1.0,
            "safe_prompt": False
        }

        if self.dryrun:
            print(f"Dry run: Simulating API call for question '{question}'")
            # Generate a fake but valid answer
            content = {
                "response": "This is a simulated response to your question."
            }
            # Simulate the structure of the actual API response
            return {"choices": [{"message": {"content": json.dumps(content)}}]}
        else:
            try:
                print(f"Sending request to API for question '{question}'...")
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                print("Request successful.")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error calling API: {e}")
                # Log more details if available in the response
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status code: {e.response.status_code}")
                    try:
                        print(f"Response text: {e.response.text}")
                    except Exception as json_err:
                        print(f"Could not decode error response JSON: {json_err}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred during API call: {e}")
                return None


    def ask(self, image_path, question):
        """Ask a question about the image using LLM API"""
        # Ensure model is Pixtral for image input
        # if not self.model.startswith("pixtral"):
        #     print(f"Warning: Model {self.model} may not support image input. Trying anyway.")
            # Or force switch: self.model = "pixtral-large-latest" 

        base64_image = self.encode_image(image_path)
        if not base64_image:
            print(f"Skipping request due to image encoding error for {image_path}")
            return None
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json", # Added for clarity
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model, 
             "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"} # Use correct MIME type
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 512, # Reduced max_tokens as the expected output is small
            "top_p": 1.0, # Adjusted top_p as per recommendations for JSON mode
            "safe_prompt": False
        }
        
        if self.dryrun:   
            print(f"Dry run: Simulating API call for image {os.path.basename(image_path)}")
            # Generate a fake but valid answer
            content = {
                "trend": "The time series shows a fluctuating trend.",
                "noise": "There appears to be moderate noise influencing the series.",
                "extrema": "Local maxima and minima are visible throughout the series, with global extrema near the start and middle."
            }
            # Simulate the structure of the actual API response
            return {"choices": [{"message": {"content": json.dumps(content)}}]} 
        else:
            try:
                print(f"Sending request to LLM API for image {os.path.basename(image_path)}...")
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                print("Request successful.")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error calling LLM API: {e}")
                # Log more details if available in the response
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status code: {e.response.status_code}")
                    try:
                        print(f"Response text: {e.response.text}")
                    except Exception as json_err:
                        print(f"Could not decode error response JSON: {json_err}")
                return None
            except Exception as e: # Catch any other unexpected errors
                print(f"An unexpected error occurred during API call: {e}")
                return None

    # generate a dataset of time series, images and description
    # the series, images and description are stored in the dataset directory
    # in the same time a json object data_json is created: it is a list of dictionnaries
    # each dictionnary has four entries: "question", "series", "image" and "description"
    # the series is a list of floats, the image is a string containing the path to the image
    # and the description is a string
    def dataset(self, n):
        directory = "dataset"
        json_file_path = os.path.join(directory, "data.jsonl")  # Change to data.jsonl

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # --- CHANGE 1: Load existing data ---
        data_json = []
        start_index = 0
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r") as f:
                    data_json = json.load(f)
                    # Ensure data_json is a list
                    if not isinstance(data_json, list):
                        print(f"Warning: {json_file_path} does not contain a JSON list. Starting fresh.")
                        data_json = []
                    else:
                        # --- CHANGE 2: Calculate starting index ---
                        start_index = len(data_json)
                        print(f"Loaded {start_index} existing entries from {json_file_path}.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {json_file_path}. Starting fresh.")
                data_json = []
            except Exception as e:
                print(f"Warning: Error reading {json_file_path}: {e}. Starting fresh.")
                data_json = []
        else:
            print(f"{json_file_path} not found. Creating a new dataset.")

        # --- Generation Loop ---
        generated_count = 0
        max_attempts_per_item = 3  # Add a limit to retry attempts for JSON validation

        for i in range(n):
            # --- CHANGE 3: Use correct index for new entries and filenames ---
            current_index = start_index + i

            print(f"Generating time series {i}/{n-1} (Overall index: {current_index})")

            # Define file paths using the unique current_index
            base_filename = f"ou_process_{current_index}"
            image_path = os.path.join(directory, f"{base_filename}.png")
            dat_path = os.path.join(directory, f"{base_filename}.dat")
            txt_path = os.path.join(directory, f"{base_filename}.txt")  # For saving raw description

            ou = OUProcess()
            ts, truth = ou.generate(image_path, dat_path)

            # Parse the truth description
            truth_dict = json.loads(truth)

            attempts = 0
            valid_json_response = False
            while attempts < max_attempts_per_item and not valid_json_response:
                attempts += 1
                print(f"  Attempting LLM description (Attempt {attempts}/{max_attempts_per_item})")
                if self.image:
                    response = self.ask(image_path, prompt_ts)
                else:
                    response = self.ask_noimage(prompt_ts)

                if response is None or "choices" not in response or not response["choices"]:
                    print("  Error: Failed to get response from LLM or response is empty.")
                    continue  # Retry if API failed

                raw_json_string = response["choices"][0]["message"]["content"]
                print(f"  Raw LLM response: {raw_json_string}")

                # Save the raw response text regardless of validity for debugging
                try:
                    with open(txt_path, "w") as f:
                        f.write(raw_json_string)
                except IOError as e:
                    print(f"  Warning: Could not write raw description to {txt_path}: {e}")

                # Check if the response is valid JSON and has the required keys
                if check_json_format(raw_json_string):
                    try:
                        # Re-parse the cleaned string if check_json_format modified it
                        cleaned_json_string = raw_json_string
                        if cleaned_json_string.strip().startswith("```json"):
                            cleaned_json_string = cleaned_json_string.strip()[7:-3].strip()
                        elif cleaned_json_string.strip().startswith("```"):
                            cleaned_json_string = cleaned_json_string.strip()[3:-3].strip()

                        description_json = json.loads(cleaned_json_string)
                        ts_list = ts.tolist()

                        data_json.append({
                            "index": current_index,
                            "question": prompt_ts,
                            "series": ts_list,
                            "image": image_path,
                            "description": description_json,
                            "truth_description": truth_dict
                        })
                        valid_json_response = True
                        generated_count += 1
                        print(f"  Successfully added entry with index {current_index}.")
                        break  # Exit the retry loop on success
                    except json.JSONDecodeError as e:
                        print(f"  Error: Could not re-parse validated JSON string: {e}. Raw string was: {raw_json_string}")
                    except Exception as e:
                        print(f"  An unexpected error occurred while processing valid response: {e}")
                else:
                    print(f"  Warning: Invalid JSON format received on attempt {attempts}. Raw response was: {raw_json_string}")
                    if attempts >= max_attempts_per_item:
                        print(f"  Error: Max attempts reached for index {current_index}. Skipping this entry.")

        # --- CHANGE 4: Save the combined data as JSON Lines ---
        try:
            with open(json_file_path, "w") as f:
                for entry in data_json:
                    json.dump(entry, f)
                    f.write('\n')
            print(f"\nSuccessfully generated {generated_count} new entries.")
            print(f"Total entries in {json_file_path}: {len(data_json)}")
        except IOError as e:
            print(f"\nError: Could not write updated dataset to {json_file_path}: {e}")
        except TypeError as e:
            print(f"\nError: Could not serialize data to JSON: {e}. Check data types.")

    def curate_dataset(self):
        """Curate the dataset by generating new entries and saving them as JSON Lines."""
        directory = "dataset"
        json_file_path = os.path.join(directory, "data.jsonl")  # Change to data.jsonl
        # load the jsonl file
        data_json = []
        try:
            with open(json_file_path, "r") as f:
                for line in f:
                    data_json.append(json.loads(line))
        except IOError as e:
            print(f"\nError: Could not read dataset from {json_file_path}: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"\nError: Could not decode JSON from {json_file_path}: {e}")
            return
        
        index_to_remove = []

        # for each entry compute NLI score
        for i, entry in enumerate(data_json):
            print(f"\nProcessing entry {i}/{len(data_json)-1}")
            description = entry["description"]
            # extract the trend
            trend = description["trend"]
            # extract the noise
            noise = description["noise"]
            # extract the extrema
            extrema = description["extrema"]
            truth_description = entry["truth_description"]
            # extract the trend
            truth_trend = truth_description["trend"]
            # extract the noise
            truth_noise = truth_description["noise"]
            # extract the extrema
            truth_extrema = truth_description["extrema"]
            # compute the NLI score for the trend
            score_trend = detect_contradiction_nli(trend, truth_trend)
            print(truth_description["parameters"])
            print("trend", trend)
            print("truth_trend", truth_trend)
            print("score_trend", score_trend)
            score_noise = detect_contradiction_nli(noise, truth_noise)
            print("noise", noise)
            print("truth_noise", truth_noise)
            print("score_noise", score_noise)
            score_extrema = detect_contradiction_nli(extrema, truth_extrema)
            print("extrema", extrema)
            print("truth_extrema", truth_extrema)
            print("score_extrema", score_extrema)
            # remove the entry from the dictionnary if the first tuple value 
            # of one of the scores is 'no' 
            if score_trend[0] == 'no' or score_noise[0] == 'no' or score_extrema[0] == 'no':
                index_to_remove.append(i)
        print("index_to_remove", index_to_remove)
        


import re

def ask_noimage(question):
    """Ask a question using text only (no image) via the LLM API, expecting plain text response."""

    qwen_api_key = os.getenv("TEXTSYNTH_API_KEY")
    qwen_url = "https://palgania.ovh:8106/v1/chat/completions"
    if not qwen_api_key:
        raise ValueError("Qwen API key not found in environment variables.")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {qwen_api_key}"
    }

    payload = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ],
        "max_tokens": 2048,
        # Removed "response_format"
    }

    try:
        print("Sending text-only request to Palgania API...")
        response = requests.post(qwen_url, headers=headers, json=payload, timeout=60, verify=False)
        response.raise_for_status()
        print("Request successful.")

        # Extract and return plain text response from LLM
        data = response.json()
        answer = data['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"Error calling Palgania API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            try:
                print(f"Response text: {e.response.text}")
            except Exception as json_err:
                print(f"Could not decode error response JSON: {json_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return None
    
    #remove the block <think> ... </think>
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    answer = answer.strip()
    print(f"LLM response: {answer}")

    return answer

            
        


if __name__ == "__main__":
    # Make sure MISTRAL_API_KEY is set as an environment variable if dryrun=False
    # Example (Linux/macOS): export MISTRAL_API_KEY='your_api_key'
    # Example (Windows CMD): set MISTRAL_API_KEY=your_api_key
    # Example (Windows PowerShell): $env:MISTRAL_API_KEY='your_api_key'
    
    # Set dryrun=False to actually call the API
    chat = Mistral(dryrun = False, image = False)
                                
    print("\n--- Running dataset generation (first call) ---")
    for itt in range(1):
        chat.dataset(10) 

    print("\n--- Running dataset generation (second call) ---")
    #chat.dataset(15) # Generate 2 *more* samples (total should be 5)

    print("\n--- Dataset generation complete ---")

    print("\n--- Curate the dataset ---")
    chat.curate_dataset()
    print("\n--- Curating complete ---")

    # Optional: Verify the final JSON file content
    json_file_path = os.path.join("dataset", "data.json")
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r") as f:
                final_data = json.load(f)
                print(f"Final number of entries in {json_file_path}: {len(final_data)}")
                # print("First entry:", json.dumps(final_data[0], indent=2) if final_data else "None")
                # print("Last entry:", json.dumps(final_data[-1], indent=2) if final_data else "None")
        except Exception as e:
            print(f"Could not read or parse final JSON file: {e}")