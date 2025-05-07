import base64
import requests
import os
import json
import matplotlib.pyplot as plt
import numpy as np


prompt_ts = r"""
Describe the time series in three sentences. First sentence: describe increasing/decreasing/flat trend. Second sentence: possible presence and intensity of noise. Third sentence: describe local and global extrema.
Put the description in a JSON format with the following pattern
{ "trend": <sentence1>,
  "noise": <sentence2>,
  "extrema": <sentence3> }.
"""


# check that a json object can be extracted from a string and that
# the json object has the keys "trend", "noise", and "extrema"
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


# a small class for generating Orstein Uhlenbeck process
class OUProcess:
    def __init__(self):
        self.seed = 0
        self.theta=0.7      # Speed of mean reversion: higher values pull X(t) faster towards the mean
        self.mu=0.0         # Long-term mean: the value to which the process tends to revert
        self.sigma=0.3      # Volatility: the intensity of the random fluctuations (Brownian motion component)
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
        
        # Apply random variations to parameters for diversity
        # Ensure parameters stay within reasonable bounds if necessary
        theta = max(0.1, self.theta + np.random.uniform(-0.5, 0.5)) # Keep theta positive
        mu = self.mu + np.random.uniform(-1,1)*2
        sigma = max(0.05, self.sigma + np.random.uniform(-0.2, 0.2)) # Keep sigma positive
        #sigma = 0 # Keep this commented unless you specifically want no noise
        x0 = self.x0 + np.random.uniform(-1,1)*2
        X[0] = x0                     # Initial value

        # Use a fixed seed for reproducibility *within a single generation* if needed
        # np.random.seed(self.seed) # Uncomment if you want deterministic generation based on seed
        
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(self.dt))  # Brownian increment
            X[i] = X[i-1] + theta * (mu - X[i-1]) * self.dt + sigma * dW
        
        # get the max of the sequence
        max_value = np.max(X)
        # get the min of the sequence
        min_value = np.min(X)
        # scale the sequence so that all the values are between 0 and 99.9999
        X = (X - min_value) / (max_value - min_value) * 99.9999
        # round the values to 0 decimal places and convert to integer
        X = np.floor(X).astype(int)
        
        # Save the process to a file
        np.savetxt(filename, X)
        
        plt.figure(figsize=(10, 5))
        plt.plot(ti, X )
        plt.title('Time Series')
        plt.xlabel('Time')
        plt.ylabel('X(t)')
        plt.grid(True)
        # now save the picture to a png file
        plt.savefig(imagename)
        plt.close() # Close the plot to free memory
        return X



# a simple class for asking questions to a LLM about a given picture
# the picture is encoded in base64 and sent to the LLM
class Mistral:
    def __init__(self, dryrun = True):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key and not dryrun:
             raise ValueError("MISTRAL_API_KEY environment variable not set.")
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        #self.model = "mistral-large-latest" # Using the recommended model for function calling / JSON mode
        self.model = "pixtral-large-latest"
        #self.model = "pixtral-12b-2409"
        #self.model = "pixtral-large-latest" # Use pixtral if image input is definitely needed later
        self.dryrun = dryrun # if True, do not send the request to the LLM
        print("Mistral initialized")
        #print(f"Mistral API key: {'Set' if self.api_key else 'Not Set'}")
        print(f"Mistral API URL: {self.api_url}")
        print(f"Mistral model: {self.model}")
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
        
    def ask(self, image_path, question):
        """Ask a question about the image using Pixtral."""
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
                print(f"Sending request to Mistral API for image {os.path.basename(image_path)}...")
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                print("Request successful.")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error calling Mistral API: {e}")
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
    # --- MODIFIED TO APPEND ---
    def dataset(self, n):
        directory = "dataset"
        json_file_path = os.path.join(directory, "data.json")
        
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
                data_json = [] # Start fresh if file is corrupted
            except Exception as e:
                print(f"Warning: Error reading {json_file_path}: {e}. Starting fresh.")
                data_json = []
        else:
            print(f"{json_file_path} not found. Creating a new dataset.")

        # --- Generation Loop ---
        generated_count = 0
        max_attempts_per_item = 3 # Add a limit to retry attempts for JSON validation

        for i in range(n):
            # --- CHANGE 3: Use correct index for new entries and filenames ---
            current_index = start_index + i 
            
            print(f"Generating time series {i+1}/{n} (Overall index: {current_index})")
            
            # Define file paths using the unique current_index
            base_filename = f"ou_process_{current_index}"
            image_path = os.path.join(directory, f"{base_filename}.png")
            dat_path = os.path.join(directory, f"{base_filename}.dat")
            txt_path = os.path.join(directory, f"{base_filename}.txt") # For saving raw description
            
            ou = OUProcess()
            ts = ou.generate(image_path, dat_path)
            
            attempts = 0
            valid_json_response = False
            while attempts < max_attempts_per_item and not valid_json_response:
                attempts += 1
                print(f"  Attempting LLM description (Attempt {attempts}/{max_attempts_per_item})")
                response = self.ask(image_path, prompt_ts)

                if response is None or "choices" not in response or not response["choices"]:
                    print("  Error: Failed to get response from LLM or response is empty.")
                    # Decide if you want to retry or break here
                    # break # Example: Stop trying for this item if API fails
                    continue # Example: Retry if API failed

                raw_json_string = response["choices"][0]["message"]["content"]
                print(f"  Raw LLM response: {raw_json_string}") # Log the raw response
                
                # Save the raw response text regardless of validity for debugging
                try:
                    with open(txt_path, "w") as f:
                        f.write(raw_json_string) # Save the raw string
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
                            "index": current_index, # Use the unique overall index
                            "question": prompt_ts,
                            "series": ts_list,
                            "image": image_path, # Store relative path
                            "description": description_json # Store the parsed JSON object
                        })
                        valid_json_response = True
                        generated_count += 1
                        print(f"  Successfully added entry with index {current_index}.")
                        break # Exit the retry loop on success
                    except json.JSONDecodeError as e:
                         print(f"  Error: Could not re-parse validated JSON string: {e}. Raw string was: {raw_json_string}")
                    except Exception as e: # Catch other unexpected errors during appending
                         print(f"  An unexpected error occurred while processing valid response: {e}")
                else:
                    print(f"  Warning: Invalid JSON format received on attempt {attempts}. Raw response was: {raw_json_string}")
                    if attempts >= max_attempts_per_item:
                        print(f"  Error: Max attempts reached for index {current_index}. Skipping this entry.")

        # --- CHANGE 4: Save the combined data ---
        try:
            with open(json_file_path, "w") as f:
                json.dump(data_json, f, indent=4)
            print(f"\nSuccessfully generated {generated_count} new entries.")
            print(f"Total entries in {json_file_path}: {len(data_json)}")
        except IOError as e:
            print(f"\nError: Could not write updated dataset to {json_file_path}: {e}")
        except TypeError as e:
             print(f"\nError: Could not serialize data to JSON: {e}. Check data types.")


if __name__ == "__main__":
    # Make sure MISTRAL_API_KEY is set as an environment variable if dryrun=False
    # Example (Linux/macOS): export MISTRAL_API_KEY='your_api_key'
    # Example (Windows CMD): set MISTRAL_API_KEY=your_api_key
    # Example (Windows PowerShell): $env:MISTRAL_API_KEY='your_api_key'
    
    # Set dryrun=False to actually call the API
    chat = Mistral(dryrun=False) 
                                
    print("\n--- Running dataset generation (first call) ---")
    for itt in range(2):
        chat.dataset(50) 

    print("\n--- Running dataset generation (second call) ---")
    #chat.dataset(15) # Generate 2 *more* samples (total should be 5)

    print("\n--- Dataset generation complete ---")

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