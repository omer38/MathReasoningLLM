import io
import sys
import openai
import torch
from torch.nn import DataParallel
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
################################################################################################################
###Question 1: Create a .env file with the api keys with the input OPENAI_API_KEY= before executing the code###
################################################################################################################
raise NotImplementedError("Create a .env file with the api keys with the input OPENAI_API_KEY= before executing the code")
client = openai.OpenAI()

def predict_gpt(openai_m, messages):
    ######################################
    ###Question 2: INSERT THE CODE HERE###
    ######################################
    raise NotImplementedError("Build the OpenAI function here based on the configurations given in the assignment")
    """
    This function, `predict_gpt`, is designed to interact with OpenAI's GPT model to generate predictions 
    based on a conversation history. 

    Args:
        openai_m: The OpenAI client object used to make API calls.
        messages: A list of dictionaries representing the conversation history, 
                  where each dictionary has a "role" (e.g., "system", "user", or "assistant") 
                  and "content" (the message text).
        configurations: the model used should be gpt-4o-mini, with temperature of 0.
        Note: Please do not change the gpt model, as higher accuracy does not lead to a higher mark in the assignment. 
        The assessment would mainly be assess the correctness of the implementation, rather than the performance 

    Returns:
        The model's response as a string.
    """


    ## In the final step, you can use client.chat.completions.create(...., model="gpt-4o-mini", ....) to create the model (complete)
    ## As mentioned above, make sure you use *only* gpt-4o-mini with your OpenAI key!

    return 

def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device=None):
    if model_type == "gpt":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_gpt(model, messages)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Model result: {model_result}")
    return model_result
