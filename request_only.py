import base64
import requests
import os
import json

def encode_image(image_path):
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

# Path to your image
image_path = "ou_process.png"

# Getting the base64 string
base64_image = encode_image(image_path)

# Retrieve the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]

# Specify model
model = "pixtral-12b-2409"

# Define the messages for the chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        ]
    }
]

# Define the API endpoint
api_url = "https://api.mistral.ai/v1/chat/completions"

# Define the headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define the payload
payload = {
    "model": model,
    "messages": messages
}

# Make the API request
response = requests.post(api_url, headers=headers, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    chat_response = response.json()
    # Print the content of the response
    print(type(chat_response))
    print(chat_response)
    print(chat_response['choices'][0]['message']['content'])
else:
    print(f"Error: Received status code {response.status_code}")
    print(response.text)
