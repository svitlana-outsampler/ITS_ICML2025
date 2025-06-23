import base64
import requests
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

prompt_ts_N_1 = r"""
/think Describe the time series in three sentences. First sentence: describe trend (increasing/decreasing/flat). Second sentence: noise intensity (low/medium/high). Third sentence: approximate localisation of global maximum (beginning/middle/end) and global minimum (beginning/middle/end).
Put the description in a JSON format with the following pattern
```json
{ "trend": <sentence1>,
  "noise": <sentence2>,
  "extrema": <sentence3> }
```
"""

prompt_ts_N_2 = r"""
/think Describe the two-time-series in **five sentences**.

**Context:**  
- You are given two time series: the **first series** is visualized in **blue**, and the **second series** in **orange**.  
- The numerical values are provided in a plain text format, each value on a separate line.  
- The file includes a header "Series 1:" followed by the full list of values for the first time series, then a second header "Series 2:" followed by the values for the second time series.  
- Each series is assumed to have the same number of values.

**Generic Pattern of the input numerical values:**  

Series 1:\n
x_1\n
x_2\n
...\n
x_N\n\n

Series 2:\n
y_1\n
y_2\n
...\n
y_N

**Instruction:**  
1. **First sentence:** Describe the **trend** (increasing / decreasing / flat) for each location (beginning / middle / end), for **each** time series.  
2. **Second sentence:** Describe the **noise intensity** (low / medium / high) for **each** time series.  
3. **Third sentence:** Give the **approximate location** of the **global maximum** and **global minimum** (beginning / middle / end) for each time series.  
4. **Fourth sentence:** State the **number of intersections** (zero / one / several) for each **location** (beginning / middle / end).  
5. **Fifth sentence:** Describe the **relative position** of the two time series (one above the other / one below the other / crossing each other) with respect to the **location** (beginning / middle / end).
Remark: call the two series "Series 1" and "Series 2" in your answer.

**Output format:**  
Return the description in **JSON** format, following this structure:
```json
{
  "trend": "<sentence1>",
  "noise": "<sentence2>",
  "extrema": "<sentence3>",
  "intersections": "<sentence4>",
  "position": "<sentence5>"
}
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

# generate the description of a time series without calling the LLM
def generate_description(X):
    n = len(X)
    

# a small class for generating Orstein Uhlenbeck process together with an automatic
# human readible description of the process
class OUProcess:
    """
    Génère 1 ou 2 processus d’Ornstein-Uhlenbeck scalés sur [0, 99].
    L’esprit des deux versions d’origine est entièrement conservé.

    Retour :
        - data      : np.ndarray shape (n_processes, N)
        - desc_json : str (description JSON sérialisée)
    """
    # ------------------------------------------------------------------ #
    def __init__(self, n_processes: int = 1, T: float = 1.0):
        if n_processes not in (1, 2):
            raise ValueError("n_processes doit valoir 1 ou 2.")
        self.n = n_processes
        self.T = T

        # Parameters for process 1
        self.theta1 = 0.7
        self.mu1 = 0.0
        self.sigma1 = 0.2
        self.x01 = 0.0
        # Parameters for process 2 (if needed)
        self.theta2 = 0.7
        self.mu2 = 0.0
        self.sigma2 = 0.2
        self.x02 = 0.0
        self.T = T

        # For backward compatibility with the rest of the code, keep lists
        if self.n == 1:
            self.theta = [self.theta1]
            self.mu = [self.mu1]
            self.sigma = [self.sigma1]
            self.x0 = [self.x01]
        else:
            self.theta = [self.theta1, self.theta2]
            self.mu = [self.mu1, self.mu2]
            self.sigma = [self.sigma1, self.sigma2]
            self.x0 = [self.x01, self.x02]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _describe_trend(x):
        m1, m2 = np.mean(x[:20]), np.mean(x[-20:])
        if m1 < m2 - 3:  return "increasing"
        if m1 > m2 + 3:  return "decreasing"
        return "flat"

    @staticmethod
    def _describe_noise(x):
        t = np.arange(len(x))
        resid = x - np.poly1d(np.polyfit(t, x, 3))(t)
        l2 = np.linalg.norm(resid) / np.sqrt(len(x))
        if l2 < 2:   return "low"
        if l2 > 12:  return "high"
        return "medium"

    @staticmethod
    def _describe_extrema(x):
        loc = lambda p: "beginning" if p < 32 else "end" if p > 96 else "middle"
        return f"global maximum: {loc(np.argmax(x))}, global minimum: {loc(np.argmin(x))}"

    # ------------------------------------------------------------------ #
    def generate(self,
                 imagename: str = "ou_process.png",
                 filename:  str = "ou_process.dat"):

        N  = 128
        dt = self.T / N
        t  = np.arange(N)

        # ----------------- Tirage aléatoire des paramètres ---------------- #
        # Randomize OU process parameters for diversity between runs
        for i in range(self.n):
            # Add random variation to theta (mean reversion speed), ensure it's positive
            self.theta[i] = max(0.1, self.theta[i] + np.random.uniform(-0.5, 0.5))
            if self.n == 2:
            # For two processes, allow different randomization for each
                if i == 0:
                    # Process 1: broader range for theta and sigma
                    self.theta[i] = max(0.1, self.theta1 + np.random.uniform(-0.5, 0.5))
                    self.sigma[i] = self.sigma1 + np.random.uniform(0., 0.4)
                else:
                    # Process 2: less volatility for theta and sigma
                    self.theta[i] = max(0.1, self.theta2 + np.random.uniform(0., 0.3))
                    self.sigma[i] = self.sigma2 + np.random.uniform(0., 0.15)
                # Randomize mean and initial value for both processes
                self.mu[i] = self.mu[i] + np.random.uniform(-1, 1)
                self.x0[i] = self.x0[i] + np.random.uniform(-1, 1)
            else:
                # For single process, randomize all parameters
                self.theta[i] = max(0.1, self.theta1 + np.random.uniform(-0.5, 0.5))
                self.sigma[i] = self.sigma1 + np.random.uniform(0., 0.4)
                self.mu[i] = self.mu[i] + np.random.uniform(-1, 1)
                self.x0[i] = self.x0[i] + np.random.uniform(-1, 1)

        # --------------------------- Simulation --------------------------- #
        data = np.zeros((self.n, N), dtype=int)

        for p in range(self.n):
            X = np.zeros(N)
            X[0] = self.x0[p]
            for k in range(1, N):
                dW = np.random.normal(0, np.sqrt(dt))
                X[k] = X[k-1] + self.theta[p] * (self.mu[p] - X[k-1]) * dt + self.sigma[p] * dW

            # Mise à l’échelle 0-99.9999 puis plancher
            X = (X - X.min()) / (X.max() - X.min()) * 99.9999
            data[p] = np.floor(X).astype(int)

        # ------------------------- Sauvegardes ---------------------------- #
        with open(filename, "w") as f:
            for p, serie in enumerate(data, 1):
                f.write(f"Series {p}:\n")
                f.writelines(f"{v}\n" for v in serie)
                f.write("\n")

        plt.figure(figsize=(10, 5))
        plt.ylim(0, 100)
        plt.gca().set_aspect('equal', adjustable='box')
        for p, serie in enumerate(data, 1):
            plt.plot(t, serie, label=f"Series {p}")
        plt.title(f"{self.n} OU process{'es' if self.n > 1 else ''}")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.savefig(imagename)
        plt.close()

        # ------------------------- Description ---------------------------- #
        # Compose the description JSON in the required flat format
        if self.n == 1:
            desc = {
            "trend": f"the time series shows an overall {self._describe_trend(data[0])} trend.",
            "noise": f"the time series presents {('many ' if self._describe_noise(data[0]) == 'medium' else '')}{self._describe_noise(data[0])} fluctuations.",
            "extrema": f"The maximum is reached {('around the beginning part' if np.argmax(data[0]) < 32 else 'towards the end' if np.argmax(data[0]) > 96 else 'around the middle')} of the time series and the minimum is reached {('around the beginning part' if np.argmin(data[0]) < 32 else 'towards the end' if np.argmin(data[0]) > 96 else 'around the middle')} of the time series.",
            "parameters": f"sigma={self.sigma[0]} mu={self.mu[0]} x0={self.x0[0]} theta={self.theta[0]}"
            }
        else:
            # n == 2
            trend1 = self._describe_trend(data[0])
            trend2 = self._describe_trend(data[1])
            noise1 = self._describe_noise(data[0])
            noise2 = self._describe_noise(data[1])
            desc = {
                "trend": f"Series 1 shows an overall {trend1} trend; Series 2 shows an overall {trend2} trend.",
                "noise": f"Series 1 presents {('many ' if noise1 == 'medium' else '')}{noise1} fluctuations; Series 2 presents {('many ' if noise2 == 'medium' else '')}{noise2} fluctuations.",
                "extrema": f"For Series 1, the maximum is reached {('around the beginning part' if np.argmax(data[0]) < 32 else 'towards the end' if np.argmax(data[0]) > 96 else 'around the middle')} and the minimum is reached {('around the beginning part' if np.argmin(data[0]) < 32 else 'towards the end' if np.argmin(data[0]) > 96 else 'around the middle')}. For Series 2, the maximum is reached {('around the beginning part' if np.argmax(data[1]) < 32 else 'towards the end' if np.argmax(data[1]) > 96 else 'around the middle')} and the minimum is reached {('around the beginning part' if np.argmin(data[1]) < 32 else 'towards the end' if np.argmin(data[1]) > 96 else 'around the middle')}.",
                # intersections and position will be added below if n==2
                "parameters": f"sigma1={self.sigma[0]} mu1={self.mu[0]} x01={self.x0[0]} theta1={self.theta[0]} sigma2={self.sigma[1]} mu2={self.mu[1]} x02={self.x0[1]} theta2={self.theta[1]}"
            }

        # Ajout des intersections / position relative si n=2
        if self.n == 2:
            diff = np.sign(data[0] - data[1])
            idx  = np.where(np.diff(diff) != 0)[0]
            intersection_count = len(idx)

            loc_map = { "beginning": (0, 32), "middle": (32, 96), "end": (96, 128) }
            loc_counts = {k: ((idx >= a) & (idx < b)).sum() for k, (a, b) in loc_map.items()}

            if intersection_count == 0:
                desc["intersections"] = "zero intersections"
            else:
                parts = [f"{v or 'zero'} in the {k}" for k, v in loc_counts.items()]
                word  = "one intersection" if intersection_count == 1 else "several intersections"
                desc["intersections"] = f"{word} ({', '.join(parts)})"

            # Position relative (majoritairement au-dessus / en dessous / mélangée)
            rel_parts, thr = [], 8
            for k, (a, b) in loc_map.items():
                segment = diff[a:b]
                n_above = np.sum(segment == 1)
                n_below = np.sum(segment == -1)
                if n_above > n_below + thr:
                    rel_parts.append(f"Series 1 is mostly above Series 2 in the {k}")
                elif n_below > n_above + thr:
                    rel_parts.append(f"Series 2 is mostly above Series 1 in the {k}")
                else:
                    gap = np.abs(data[0, a:b] - data[1, a:b])
                    if gap.mean() > 0.5 * gap.std():
                        rel_parts.append(f"the series are mostly separated in the {k}")
                    else:
                        rel_parts.append(f"the series are mostly overlapping in the {k}")
            desc["position"] = "; ".join(rel_parts)

        return data, json.dumps(desc, ensure_ascii=False, indent=2)
    
# recompute the truth 
def ask_truth(X):
    x = np.arange(len(X))
    coeffs = np.polyfit(x, X, deg=3)
    P = np.poly1d(coeffs)
    X_fit = P(x)
    # compute the l2 norm of the difference
    l2_norm = np.linalg.norm(X - X_fit)/np.sqrt(len(X))
    # plt.plot(x,X)
    # plt.plot(x,X_fit)
    # plt.show()
    Pp = P.deriv()
    Xp_fit = Pp(x)
    delta = Xp_fit[-1] - Xp_fit[0]
    # divmin = np.min(Xp_fit)
    # divmax = np.max(Xp_fit)
    # if divmin > 0:
    #     sentence =  "the time series presents an overall increasing trend"
    # elif divmax < 0:
    #     sentence = "the time series presents an overall decreasing trend"
    # else:
    #     sentence = "the time series presents no uniformly increasing or decreasing trend"
    # if delta > 5:
    #     sentence =  "the time series presents an overall increasing trend"
    # elif delta < -5:
    #     sentence = "the time series presents an overall decreasing trend"
    # else:
    #     sentence = "the time series presents no uniformly increasing or decreasing trend"
    # compute the average of the solution on the 20 first points
    average1 = np.mean(X[:20])
    # compute the average of the solution on the 20 last points
    average2 = np.mean(X[-20:])
    if average1 < average2 -3:
        sentence_trend = "the time series shows an overall increasing trend."
    elif average1 > average2 +3:
        sentence_trend = "the time series shows an overall decreasing trend."
    else:
        sentence_trend = "the time series shows no uniformly increasing or decreasing trend."

    print(f"L2 norm of the difference: {l2_norm}")
    #sentence_noise = self.data_json[i]["description"]["noise"]
    if l2_norm < 2:
        sentence_noise = "the noise intensity is low"
    elif l2_norm > 12:
        sentence_noise = "the noise intensity is high"
    else:
        sentence_noise = "the noise intensity is medium"
    # print(sentence_noise)
    # self.data_json[i]["truth_description"]["noise"] = sentence_noise
    # recherche de la localisation en t du maximum et du minimum
    pos_max = np.argmax(X)
    print('pos_max', pos_max)
    if pos_max < 32:
        sentence_extrema = "The maximum is reached around the beginning part of the time series"
    elif pos_max > 96:
        sentence_extrema = "The maximum is reached towards the end of the time series"
    else:
        sentence_extrema = "The maximum is reached around the middle of the time series"

    pos_min = np.argmin(X)
    print('pos_min', pos_min)
    if pos_min < 32:
        sentence_extrema += " and the minimum is reached around the beginning part of the time series."
    elif pos_min > 96:
        sentence_extrema += " and the minimum is reached towards the end of the time series."
    else:
        sentence_extrema += " and the minimum is reached around the middle of the time series."

    return (sentence_trend, sentence_noise, sentence_extrema)


# définir les modèles: pixtral, mistral, qwen, nollm (génération de phrases factuelles)
# appeler ask ou ask noimage en fonction du modèle
# fact checker 

# a simple class for asking questions to a LLM about a given picture
# the picture is encoded in base64 and sent to the LLM
class Mistral:
    def __init__(self, dryrun = False, image = False, n_processes = 1):
        self.dryrun = dryrun
        self.image = image
        self.n_processes = n_processes
        if image:
            self.api_key = os.environ.get("MISTRAL_API_KEY")
            if not self.api_key and not dryrun:
                raise ValueError("MISTRAL_API_KEY environment variable not set.")
            self.api_url = "https://api.mistral.ai/v1/chat/completions"
            #self.model = "mistral-large-latest" # Using the recommended model for function calling / JSON mode
            self.model = "pixtral-large-latest"
            #self.model = "pixtral-12b-2409"
            #self.model = "pixtral-large-latest" # Use pixtral if image input is definitely needed later
            print("Mistral initialized")
        else:
            self.api_key = os.environ.get("TEXTSYNTH_API_KEY")
            # if not self.api_key and not dryrun:
            #     raise ValueError("TEXTSYNTH_API_KEY environment variable not set.")
            self.api_url = "https://palgania.ovh:8106/v1/chat/completions"
            self.api_url = "http://127.0.0.1:8081/v1/chat/completions"
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
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=900)
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
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=900) # Added timeout
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
            
    def load_dataset(self):
        directory = "dataset"
        json_file_path = os.path.join(directory, "data.jsonl")  # Change to data.jsonl

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # --- CHANGE 1: Load existing data ---
        self.data_json = []
        start_index = 0
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r") as f:
                    self.data_json = []
                    for line in f:
                        try:
                            entry = json.loads(line)
                            self.data_json.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not decode JSON from line: {line}. Error: {e}")
                    # Ensure data_json is a list
                    if not isinstance(self.data_json, list):
                        print(f"Warning: {json_file_path} does not contain valid JSON lines. Starting fresh.")
                        self.data_json = []
                    else:
                        # --- CHANGE 2: Calculate starting index ---
                        start_index = len(self.data_json)
                        print(f"Loaded {start_index} existing entries from {json_file_path}.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {json_file_path}. Starting fresh.")
                self.data_json = []
            except Exception as e:
                print(f"Warning: Error reading {json_file_path}: {e}. Starting fresh.")
                self.data_json = []
        else:
            print(f"{json_file_path} not found. Creating a new dataset.")
            # just create an empty file
            with open(json_file_path, 'w') as file:
                pass

    def save_dataset(self):
        directory = "dataset"
        if not os.path.exists(directory):
            os.makedirs(directory)
        json_file_path = os.path.join(directory, 'data2.jsonl')
        with open(json_file_path, 'w') as file:
            for entry in self.data_json:
                file.write(json.dumps(entry) + '\n')

    # recompute the truth 
    def redo_truth(self):
        self.load_dataset()
        np.random
        for i in range(len(self.data_json)):
            X = np.array(self.data_json[i]["series"])
            x = np.arange(len(X))
            coeffs = np.polyfit(x, X, deg=3)
            P = np.poly1d(coeffs)
            X_fit = P(x)
            # compute the l2 norm of the difference
            l2_norm = np.linalg.norm(X - X_fit)/np.sqrt(len(X))
            # plt.plot(x,X)
            # plt.plot(x,X_fit)
            # plt.show()
            Pp = P.deriv()
            Xp_fit = Pp(x)
            delta = Xp_fit[-1] - Xp_fit[0]
            # divmin = np.min(Xp_fit)
            # divmax = np.max(Xp_fit)
            # if divmin > 0:
            #     sentence =  "the time series presents an overall increasing trend"
            # elif divmax < 0:
            #     sentence = "the time series presents an overall decreasing trend"
            # else:
            #     sentence = "the time series presents no uniformly increasing or decreasing trend"
            # if delta > 5:
            #     sentence =  "the time series presents an overall increasing trend"
            # elif delta < -5:
            #     sentence = "the time series presents an overall decreasing trend"
            # else:
            #     sentence = "the time series presents no uniformly increasing or decreasing trend"
            # compute the average of the solution on the 20 first points
            average1 = np.mean(X[:20])
            # compute the average of the solution on the 20 last points
            average2 = np.mean(X[-20:])
            if average1 < average2 -3:
                sentence = "the time series shows an overall increasing trend."
            elif average1 > average2 +3:
                sentence = "the time series shows an overall decreasing trend."
            else:
                sentence = "the time series shows no uniformly increasing or decreasing trend."
            self.data_json[i]["truth_description"]["trend"] = sentence

            print(f"{i} L2 norm of the difference: {l2_norm}")
            #sentence_noise = self.data_json[i]["description"]["noise"]
            if l2_norm < 2:
                sentence_noise = "the noise intensity is low"
            elif l2_norm > 12:
                sentence_noise = "the noise intensity is high"
            else:
                sentence_noise = "the noise intensity is medium"
            print(sentence_noise)
            self.data_json[i]["truth_description"]["noise"] = sentence_noise
        self.save_dataset()

            


    # generate a dataset of time series, images and description
    # the series, images and description are stored in the dataset directory
    # in the same time a json object data_json is created: it is a list of dictionnaries
    # each dictionnary has four entries: "question", "series", "image" and "description"
    # the series is a list of floats, the image is a string containing the path to the image
    # and the description is a string
    def dataset(self, n):
        self.load_dataset()  # Load existing dataset if any
        start_index = len(self.data_json)
        directory = "dataset"
        # --- Generation Loop ---
        generated_count = 0
        max_attempts_per_item = 3  # Add a limit to retry attempts for JSON validation
        n_processes = self.n_processes

        for i in range(n):
            # --- CHANGE 3: Use correct index for new entries and filenames ---
            current_index = start_index + i

            print(f"Generating time series {i}/{n-1} (Overall index: {current_index})")

            # Define file paths using the unique current_index
            base_filename = f"ou_process_{current_index}"
            image_path = os.path.join(directory, f"{base_filename}.png")
            dat_path = os.path.join(directory, f"{base_filename}.dat")
            txt_path = os.path.join(directory, f"{base_filename}.txt")  # For saving raw description

            ou = OUProcess(n_processes=n_processes, T=1.0)
            ts, truth = ou.generate(image_path, dat_path)

            # Parse the truth description
            truth_dict = json.loads(truth)

            attempts = 0
            valid_json_response = False
            while attempts < max_attempts_per_item and not valid_json_response:
                attempts += 1
                print(f"  Attempting LLM description (Attempt {attempts}/{max_attempts_per_item})")
                prompt_ts = prompt_ts_N_2 if self.n_processes == 2 else prompt_ts_N_1
                if self.image:
                    response = self.ask(image_path, prompt_ts)
                else:
                    response = self.ask_noimage(prompt_ts)

                if response is None or "choices" not in response or not response["choices"]:
                    print("  Error: Failed to get response from LLM or response is empty.")
                    continue  # Retry if API failed

                raw_json_string = response["choices"][0]["message"]["content"]
                print(f"  Raw LLM response: {raw_json_string}")
                # remove the think block 
                raw_json_string = re.sub(r'<think>.*?</think>', '', raw_json_string, flags=re.DOTALL)
                raw_json_string = raw_json_string.strip()


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

                        self.data_json.append({
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
        directory = "dataset"
        json_file_path = os.path.join(directory, "data.jsonl")
        try:
            with open(json_file_path, "w") as f:
                for entry in self.data_json:
                    json.dump(entry, f)
                    f.write('\n')
            print(f"\nSuccessfully generated {generated_count} new entries.")
            print(f"Total entries in {json_file_path}: {len(self.data_json)}")
        except IOError as e:
            print(f"\nError: Could not write updated dataset to {json_file_path}: {e}")
        except TypeError as e:
            print(f"\nError: Could not serialize data to JSON: {e}. Check data types.")

    def curate_dataset(self):
        """Curate the dataset by generating new entries and saving them as JSON Lines."""
        self.load_dataset()        
        index_to_remove = []

        errors = [0,0,0]

        # for each entry compute NLI score
        for i, entry in enumerate(self.data_json):
            print(f"\nProcessing entry {i}/{len(self.data_json)-1}")
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
                if score_trend[0] == 'no':
                    errors[0] += 1
                if score_noise[0] == 'no':
                    errors[1] += 1
                if score_extrema[0] == 'no':
                    errors[2] += 1
                    self.data_json[i]["description"]["extrema"]= self.data_json[i]["truth_description"]["extrema"]

            if self.n_processes == 2:
                # check the intersections and position
                if "intersections" in description:
                    intersections = description["intersections"]
                    truth_intersections = truth_description["intersections"]
                    score_intersections = detect_contradiction_nli(intersections, truth_intersections)
                    print("intersections", intersections)
                    print("truth_intersections", truth_intersections)
                    print("score_intersections", score_intersections)
                    if score_intersections[0] == 'no':
                        index_to_remove.append(i)
                        errors[2] += 1
                        self.data_json[i]["description"]["intersections"] = self.data_json[i]["truth_description"]["intersections"]

                if "position" in description:
                    position = description["position"]
                    truth_position = truth_description["position"]
                    score_position = detect_contradiction_nli(position, truth_position)
                    print("position", position)
                    print("truth_position", truth_position)
                    print("score_position", score_position)
                    if score_position[0] == 'no':
                        index_to_remove.append(i)
                        errors[2] += 1
                        self.data_json[i]["description"]["position"] = self.data_json[i]["truth_description"]["position"]

        print("errors", errors, "/", len(self.data_json))

        print("index_to_check", index_to_remove)
        self.save_dataset()
        


import re

def ask_noimage(question):
    """Ask a question using text only (no image) via the LLM API, expecting plain text response."""

    # qwen_api_key = os.getenv("TEXTSYNTH_API_KEY")
    # qwen_url = "https://palgania.ovh:8106/v1/chat/completions"
    # if not qwen_api_key:
    #     raise ValueError("Qwen API key not found in environment variables.")
    qwen_api_key = ""
    qwen_url= "http://127.0.0.1:8081/v1/chat/completions"
    
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
                    {"type": "text", "text": question + " /think"}
                ]
            }
        ],
        "max_tokens": 512,
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
        # remove the data in the think region
        print(f"LLM response: {answer}")
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = answer.strip()

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
    chat = Mistral(dryrun = False, image = True, n_processes = 2)
                                
    print("\n--- Running dataset generation (first call) ---")
    for itt in range(1):
        chat.dataset(1) 

    print("\n--- Running dataset generation (second call) ---")
    #chat.dataset(15) # Generate 2 *more* samples (total should be 5)

    print("\n--- Dataset generation complete ---")

    print("\n--- Curate the dataset ---")
    chat.curate_dataset()
    print("\n--- Curating complete ---")

    #chat.redo_truth()
    print("\n--- Redo truth complete ---")

    #chat.save_dataset()
    print("\n--- Dataset saved ---")

    # Optional: Verify the final JSON file content
    json_file_path = os.path.join("dataset", "data.jsonl")
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r") as f:
                final_data = [json.loads(line) for line in f]
                print(f"Final number of entries in {json_file_path}: {len(final_data)}")
                # print("First entry:", json.dumps(final_data[0], indent=2) if final_data else "None")
                # print("Last entry:", json.dumps(final_data[-1], indent=2) if final_data else "None")
        except Exception as e:
            print(f"Could not read or parse final jsonl file: {e}")