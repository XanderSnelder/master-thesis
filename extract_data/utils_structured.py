import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
import re
import time
from tqdm import tqdm
import ast
import json
import io
import numpy as np
from numpy.linalg import norm
from pydantic import BaseModel, Field
from typing import Optional


class SimilarityScore(BaseModel):
    """Pydantic model for structured similarity output"""
    word1: str = Field(description="The first word in the pair")
    word2: str = Field(description="The second word in the pair")
    score: float = Field(description="Semantic similarity score from 0 to 10 with two decimals", ge=0, le=10)

def get_response_format_schema():
    """Get the structured output schema for OpenRouter"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "similarity_response",
            "schema": SimilarityScore.model_json_schema(),
            "strict": True
        }
    }
def get_responses_single_structured(prompt, chunks, model, sample_size, delay, client):

    # List to store responses
    responses = [] 

    # Start timing total processing
    start_time_total = time.time()

    # Set up progress bar
    total_iterations = len(chunks) * sample_size
    pbar = tqdm(total=total_iterations, desc="Processing", unit="chunk")

    # Get response format
    response_format = get_response_format_schema()

    # Collect responses for each chunk and sample
    for chunk in chunks:
        for _ in range(sample_size):
            # Extract word pair from chunk
            word_pair = chunk[0]

            # Format prompts
            formatted_message = prompt.format(word1=word_pair[0], word2=word_pair[1])
            messages = [{"role": "user", "content": formatted_message}]

            # Build API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "response_format": response_format,
            }

            try:
                # Call API
                completion = client.chat.completions.create(**api_params)

                # Store response content
                responses.append(completion.choices[0].message.content)

            except Exception as e:
                print(f"\nError processing pair ({word_pair[0]}, {word_pair[1]}): {e}")
                responses.append(None)

            # Update progress bar after each sample
            pbar.update(1)

            # Delay after each API call
            time.sleep(delay)

    # Close progress bar
    pbar.close()

    # Total processing time
    end_time_total = time.time()
    print(f"Total time taken: {end_time_total - start_time_total:.2f} seconds")

    return responses
def process_responses_structured(responses):
    """
    Process structured JSON responses from the model
    """
    data_dict = {}
    
    for response in responses:
        if response is None:
            continue
        
        try:
            # Try to parse as JSON first
            parsed_response = json.loads(response)
            
            # Validate against Pydantic model
            validated = SimilarityScore.model_validate(parsed_response)
            
            key = (validated.word1, validated.word2)
            
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(float(validated.score))
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse response: {e}")
            print(f"Response content: {response[:100]}...")
            continue
    
    return data_dict