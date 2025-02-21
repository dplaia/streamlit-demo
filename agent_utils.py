import os
import time
import pickle
from os.path import join, exists, basename
from os import listdir, makedirs
from datetime import datetime
from google import genai
from google.genai import types
from openai import OpenAI, AsyncOpenAI
import requests
import json
from pydantic import BaseModel, Field
from crawl4ai import *
from pydantic_ai.result import RunResult
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from queue import Queue, Empty
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Dict, Optional, List
from typing import TypeVar, Generic
from markitdown import MarkItDown
import asyncio
#import nest_asyncio 
# Add this line to allow nested event loops
#nest_asyncio.apply()
import copy
import re
from collections import deque
from functools import wraps
import threading

from config import Config

# from loguru import logger
# import logfire

console = Console()
config = Config()

def cprint(text: str, markdown: bool = True):
    if markdown:
        console.print(Markdown(text))
    else:
        console.print(text)

def save_data(data, name):

    if not exists("temp_data"):
        makedirs("temp_data")
        
    dump_folder = "temp_data/"
    file_path = join(dump_folder, name + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_data(name):
    dump_folder = "temp_data/"
    file_path = join(dump_folder, name + ".pkl")

    if exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        return data
    
    return None

class RateLimiter:
    def __init__(self, rpm: int = 10, window: float = 60.0):
        self.rpm = rpm
        self.window = window
        self.timestamps = deque(maxlen=rpm)
        self.lock = threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                if len(self.timestamps) >= self.rpm:
                    oldest = self.timestamps[0]
                    elapsed = now - oldest
                    if elapsed < self.window:
                        sleep_time = self.window - elapsed
                        time.sleep(sleep_time)
                        now = time.time()
                self.timestamps.append(now)
            return func(*args, **kwargs)
        return wrapper

def apply_rate_limit(functions, agent_name, rpm_value):
    rate_limiter = RateLimiter(rpm=rpm_value) # Create a rate limiter instance
    return [rate_limiter(func) for func in functions]

class AsyncFunctionCallLimiter:
    def __init__(self, num: int):
        """
        Initializes the AsyncFunctionCallLimiter for async functions.

        Args:
            num: The maximum number of times EACH decorated async function can be called in total.
        """
        self.num = num
        self.function_call_counts = {}  # Dictionary to track call counts per function
        self.lock = asyncio.Lock()

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__  # Get the function name

            async with self.lock:
                if func_name not in self.function_call_counts:
                    self.function_call_counts[func_name] = 0  # Initialize count if not seen before

                if self.function_call_counts[func_name] < self.num:
                    self.function_call_counts[func_name] += 1
                    print(f"Call count (async) for '{func_name}': {self.function_call_counts[func_name]}/{self.num}")
                    return await func(*args, **kwargs)
                else:
                    return f"""Function call limit ({self.num}) reached for tool/function '{func.__name__}'. 
                    Please don't call this function again (you will get the same response every time.)."""

        return wrapper

# def scrubbing_callback(m: logfire.ScrubMatch):
#     if (
#         m.path == ('message', 'prompt')
#         and m.pattern_match.group(0) == 'Auth'
#     ):
#         return m.value

#     if (
#         m.path == ('attributes', 'prompt')
#         and m.pattern_match.group(0) == 'Auth'
#     ):
#         return m.value

#     if m.path == ('attributes', 'agent', 'model', 'auth'):
#         return m.value

def get_system_prompt(name: str) -> Optional[str]:
    if not name.endswith(".txt"):
        name = f"{name}.txt"

    directory = 'system_prompts'
    filepath = join(directory, name)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt
    else:
        print(f"File {name} does not exists in folder {directory}.")
    return None

def is_pdf_url(url: str, timeout: int = config.REQUEST_TIMEOUT) -> bool:
    """
    Determines if a URL points to a PDF document
    
    Args:
        url: Web address to check
        timeout: Request timeout in seconds
        
    Returns:
        True if content is PDF, False otherwise
        
    Raises:
        ValueError: For invalid URL format
    """
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL format")
        
    try:
        # Try HEAD first to avoid downloading content
        response = requests.head(url, allow_redirects=True, timeout=timeout)

        # Fallback to GET if HEAD not allowed
        if response.status_code == 405:
            response = requests.get(url, stream=True, timeout=timeout)

        content_type = response.headers.get('Content-Type', '').lower()
        return 'application/pdf' in content_type

    except RequestException as e:
        print(f"Error checking PDF URL: {str(e)}")
        return False
# Pre-compile pattern for better performance
WORD_PATTERN = re.compile(r"(?<!\w)'|'(?!\w)", re.UNICODE)

def word_count(text: str) -> int:
    """Count words in text using more robust word matching.
    
    Handles:
    - Contractions (don't → 1 word)
    - Hyphenated words (state-of-the-art → 1 word)
    - Apostrophes in quotes ("rock 'n' roll" → 3 words)
    """
    if not text.strip():
        return 0

    # Normalize apostrophes and hyphens
    text = WORD_PATTERN.sub('', text.replace('-', ' '))
    return len(re.findall(r"[\w']+", text))

def test_word_count():
    # Basic cases
    assert word_count("hello world") == 2
    assert word_count("Don't panic!") == 2  # Contraction
    
    # Edge cases
    assert word_count("state-of-the-art") == 1  # Hyphenated
    assert word_count("hello_world") == 1  # Underscores
    assert word_count("") == 0  # Empty string
    assert word_count("   ") == 0  # Whitespace
    
    # Punctuation handling
    assert word_count("Hello, world!") == 2
    assert word_count("Rock 'n' roll") == 3  # Apostrophes in quotes


