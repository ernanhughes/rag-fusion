import requests
import json
import logging
from config import appConfig
import ollama

logger = logging.getLogger(__name__)

def to_embedding(text, model_name = "mxbai-embed-large"):
    response = ollama.embed(model=model_name, input=text)
    return response["embeddings"]


def generate_embeddings(text, model_name=appConfig["EMBEDDING_MODEL_NAME"]):
    try:
        url = f"{appConfig['OLLAMA_BASE_URL']}/api/embed"
        data = {"input": text, "model": model_name}
        response = requests.post(url, json=data)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        embeddings = response.json()
        return embeddings["embeddings"]
    except requests.exceptions.RequestException as e: # Catch connection and HTTP errors
        logging.error(f"Failed to generate embeddings: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e: # Catch JSON errors
        logging.error(f"Failed to parse JSON response: {e}")
        return None
    
   
def chat(prompt, model_name=appConfig["CHAT_MODEL_NAME"],
         ollama_base_url=appConfig["OLLAMA_BASE_URL"]):
    """Chat with Ollama."""
    try:
        url = f"{ollama_base_url}/api/generate"
        data = {
            "prompt": prompt,
            "model": model_name,
            "stream": False
        }
        response = requests.post(url, json=data)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            response_json = response.json()
            print("Chat Response:")
            pretty_json = json.dumps(response_json, indent=4)
            logging.info(pretty_json)
            result = response_json["response"]
            print(f"For prompt: {prompt}\n result: {result}")
            return response_json["response"]
        else:
            print(f"Failed to get response from ollama. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None