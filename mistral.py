import base64
import requests
import os
import json
import matplotlib.pyplot as plt
import numpy as np


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
        N = int(self.T / self.dt)               # Number of time steps
        t = np.linspace(0, self.T, N)      # Time grid
        X = np.zeros(N)               # Array to store the process values
        theta = self.theta + np.random.uniform(-1,1)*2
        mu = self.mu + np.random.uniform(-1,1)*2
        sigma = self.sigma + np.random.uniform(-1,1)*0.25
        #sigma = 0
        x0 = self.x0 + np.random.uniform(-1,1)*2
        X[0] = x0                     # Initial value

        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(self.dt))  # Brownian increment
            X[i] = X[i-1] + theta * (mu - X[i-1]) * self.dt + sigma * dW
        
        # Save the process to a file
        np.savetxt(filename, X)
        
        plt.figure(figsize=(10, 5))
        plt.plot(t, X )
        plt.title('Time Series')
        plt.xlabel('Time')
        plt.ylabel('X(t)')
        plt.grid(True)
        # now save the picture to a png file
        plt.savefig(imagename)
        plt.close()
        return X



# a simple class for asking questions to a LLM about a given picture
# the picture is encoded in base64 and sent to the LLM
class Mistral:
    def __init__(self, dryrun = True):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.model = "pixtral-large-latest"
        #self.model = "pixtral-12b-2409"
        self.dryrun = dryrun # if True, do not send the request to the LLM
        print("Mistral initialized")
        #print(f"Mistral API key: {self.api_key}")
        print(f"Mistral API URL: {self.api_url}")
        print(f"Mistral model: {self.model}")
    
    def encode_image(self, image_path):
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:  # Added general exception handling
            print(f"Error: {e}")
            return None
        
    def ask(self, image_path, question):
        """Ask a question about the image."""
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return None
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95,
            "safe_prompt": False
        }
        if self.dryrun:   
            #print(json.dumps(payload, indent=4))
            # generate a fake but valid answer
            return {"choices": [{"message": {"content": "This is a fake answer."}}]}
        else:
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
            
        
    # generate a dataset of time series, images and description
    # the series, images and description are stored in the dataset directory
    # in the same time a json object data_json is created: it is a list of dictionnaries
    # each dictionnary has four entries: "question", "series", "image" and "description"
    # the series is a list of floats, the image is a string containing the path to the image
    # and the description is a string
    def dataset(self, n):
        directory = "dataset"
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
                    
        data_json = []

        for i in range(n):
            print(f"Generating time series {i+1}/{n}")
            ou = OUProcess()
            ts = ou.generate(
                os.path.join(directory, f"ou_process_{i}.png"),
                os.path.join(directory, f"ou_process_{i}.dat")
            )
            response = self.ask(
                os.path.join(directory, f"ou_process_{i}.png"),
                "Describe the time series in three sentences. "
                "First sentence: describe increasing/decreasing/flat pattern. "
                "Second sentence: describe the overall trend and the noise. "
                "Third sentence: describe local and globe extrema."
            )
            with open(os.path.join(directory, f"ou_process_{i}.txt"), "w") as f:
                f.write(response["choices"][0]["message"]["content"])
                f.write("\n")
            
            ts_list = ts.tolist()

            data_json.append({
                "question": "Describe the time series in three sentences. First sentence: describe increasing/decreasing/flat pattern. Second sentence: describe the overall trend and the noise. Third sentence: describe local and globe extrema.",
                "series": ts_list,
                "image": os.path.join(directory, f"ou_process_{i}.png"),
                "description": response["choices"][0]["message"]["content"]
            })
        
        with open(os.path.join(directory, "data.json"), "w") as f:
            json.dump(data_json, f, indent=4)
            

    


if __name__ == "__main__":
    chat = Mistral(dryrun = False)
    # response = chat.ask(
    #     "ou_process.png",
    #     "Describe the time series in three sentences. "
    #     "First sentence: describe increasing/decreasing/flat pattern. "
    #     "Second sentence: describe the overall trend and the noise. "
    #     "Third sentence: describe local and globe extrema."
    # )
    # print(response)

    #ou = OUProcess()
    #ou.generate("toto.png", "toto.txt")

    chat.dataset(200)