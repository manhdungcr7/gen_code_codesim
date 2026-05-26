import os
import requests
import base64
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

import os
from openai import OpenAI, AzureOpenAI
import time

usage_log_file_path = "usage_log.csv"
api_type = os.getenv("API_TYPE")

if api_type == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_URL")
elif api_type == "azure":
    api_key = os.getenv("AZURE_API_KEY")
    api_base = os.getenv("AZURE_API_URL")
    api_version = os.getenv("AZURE_API_VERSION")


class OpenAIModel(BaseModel):
    def __init__(
            self, 
            **kwargs
        ):
        pass

    def prompt(
            self, 
            processed_input: list[dict], 
            frequency_penalty=0, 
            presence_penalty=0
        ):
        pass


class OpenAIV1Model(OpenAIModel):
    def __init__(self, model_name, sleep_time=0, **kwargs):
        
        if model_name is None:
            raise Exception("Model name is required")
        
        if api_type == "azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
                timeout=120.0,
                max_retries=2
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                timeout=120.0,
                max_retries=2
            )

        self.model_name = model_name

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 4096)

        self.sleep_time = sleep_time
    

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self, 
        processed_input: list[dict], 
        frequency_penalty=0, 
        presence_penalty=0
    ):
        
        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        if self.model_name == "o3-mini" or self.model_name == "o1":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=processed_input,
                max_completion_tokens=self.max_tokens,
                stop=None,
                stream=False
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=processed_input,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=None,
                stream=False
            )

        end_time = time.perf_counter()

        with open(usage_log_file_path, mode="a") as file:
            file.write(f'{self.model_name},{response.usage.prompt_tokens},{response.usage.completion_tokens}\n')
        
        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,

            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response.choices[0].message.content,                    
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response.choices[0].message.content, run_details 

# This class is intended for only azure openai api for some special cases
# Do not use this class for openai api
class OpenAIV2Model(OpenAIModel):
    def __init__(self, model_name, sleep_time=60, **kwargs):
        if model_name is None:
            raise Exception("Model name is required")
        
        self.model_name = model_name

        self.headers = {
            "Content-Type": "application/json",
            "api-key": kwargs.get("api-key", api_key),
        }
        self.end_point = kwargs.get("end_point", api_base)

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 4096)

        self.sleep_time = sleep_time


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self, 
        processed_input: list[dict], 
        frequency_penalty=0, 
        presence_penalty=0
    ):

        time.sleep(self.sleep_time)


        # Payload for the request
        payload = {
            "messages": processed_input,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        start_time = time.perf_counter()

        response = requests.post(self.end_point, headers=self.headers, json=payload)
        # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        end_time = time.perf_counter()

        # Handle the response as needed (e.g., print or process)
        response = response.json()

        with open(usage_log_file_path, mode="a") as file:
            file.write(f'{self.model_name},{response["usage"]["prompt_tokens"]},{response["usage"]["completion_tokens"]}\n')

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],

            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response["choices"][0]["message"]["content"],                    
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response["choices"][0]["message"]["content"], run_details

