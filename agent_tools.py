import os
from enum import StrEnum
from typing import Optional, TypedDict, List, Type, TypeVar, Generic
import httpx
from config import Config
import requests
import json
#from firecrawl import FirecrawlApp
from agent_utils import *
from pydantic import BaseModel, Field
from pydantic_ai.result import RunResult
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models import Model
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from google import genai
from google.genai import types
from openai import OpenAI, AsyncOpenAI

from crawl4ai import *
import aiohttp
from bs4 import BeautifulSoup

config = Config()

# BaseAgent without dependencies, simple to use
class BasicAgent():
    def __init__(self, result_type = str, system_prompt: str = "", model: Model = None):
        if not model:
            model = GeminiModel(config.BASEAGENT_MODEL)

        self.agent = Agent(
            model,
            result_type=result_type,
            system_prompt=system_prompt)

    async def __call__(self, user_input) -> RunResult:
        return await self.agent.run(user_input)

    async def run(self, user_input) -> RunResult:
        return await self.agent.run(user_input)

class ReasoningModelAsync:
    """
    A class to handle chats using the genai Client and Chat objects.
    """
    
    def __init__(self, api_key: str = None, model: Model = None):
        """
        Initializes the ChatHandler with the given API key and model.
        
        :param api_key: Your Gemini API key.
        :param model: The model identifier to be used for the chat.
        """

        if not model:
            model_name = config.FLASH2T_MODEL
            api_key = config.GEMINI_API_KEY
        else:
            model_name = model.model_name

        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.aio.chats.create(model=model_name) # async
    
    async def __call__(self, question: str) -> str:
        """
        Sends a question to the chat and returns the text response.
        
        :param question: The question you want to ask.
        :return: The text response from the chat.
        """
        response = await self.chat.send_message(question)
        return response.text

class ReasoningModel:
    """
    A class to handle chats using the genai Client and Chat objects.
    """
    
    def __init__(self, api_key: str = None, model: Model = None):
        """
        Initializes the ChatHandler with the given API key and model.
        
        :param api_key: Your Gemini API key.
        :param model: The model identifier to be used for the chat.
        """

        if not model:
            model_name = config.FLASH2T_MODEL
            api_key = config.GEMINI_API_KEY
        else:
            model_name = model.model_name

        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(model=model_name) # sync
    
    def __call__(self, question: str) -> str:
        """
        Sends a question to the chat and returns the text response.
        
        :param question: The question you want to ask.
        :return: The text response from the chat.
        """
        response = self.chat.send_message(question)

        return response.text
    
class BasicSearchModel:
    """
    A class to handle basic websearch usinig the google grounding tool.
    """
    
    def __init__(self, perplexity_search: bool = False):
        """
        Initializes the BasicSearchAgent.

        Args:
        - perplexity_search (includes Perplexity search results)
        """

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.perplexity = perplexity_search
        

    def __call__(self, query: str):
        """
        Sends a search query and returns the search results in form of text.
        
        :param query: The query you want to search for.
        :return: The text response from the chat.
        """
        import datetime
        now = datetime.datetime.now()
        date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")

        modified_query = f"""Current Date/time: {date_time_string}
        
        Search Query:
        {query}"""

        response = self.client.models.generate_content(
            model=config.FLASH2_MODEL,
            contents=modified_query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    google_search=types.GoogleSearchRetrieval
                )]
            )
        )

        # Build the text response from the JSON response
        main_response = response.candidates[0].content.parts[0].text
        output_text = f"{main_response}"

        grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
        grounding_supports = response.candidates[0].grounding_metadata.grounding_supports

        for grounding_support in grounding_supports:
            #confidence_scores = grounding_support.confidence_scores
            indices = grounding_support.grounding_chunk_indices
            segment = grounding_support.segment
            #end_index = segment.end_index
            segment_text = segment.text

            new_segment_text = f"{segment_text} "

            for k in indices:
                new_segment_text += f"[{k+1}]"

            output_text = output_text.replace(segment_text, new_segment_text)

        if len(grounding_chunks) > 0:

            output_text += f"""
            
            References:
            
            """

            for k, grounding_chunk in enumerate(grounding_chunks):
                output_text += f"[{k+1}] {grounding_chunk.web.title} {grounding_chunk.web.uri}\n"

        if self.perplexity:
            perplexity_results = perplexity_sonar_reasoning(query)

            citation_text = ""
            for k, url in enumerate(perplexity_results['citations']):
                citation_text += f"[{k}] {url}\n"

            output_text = f"""Google Search Results: 

            {output_text}
            
            --------------------------
            Perplexity Search Results:

            {perplexity_results['text_response']}

            References:

            {citation_text}

            """
        return output_text

async def count_tokens(content: str, model_name: str):
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = await client.aio.models.count_tokens(
        model=model_name,
        contents=content,
    )
    print(response)

class TimeSpan(StrEnum):
    """
    Time span specification for Google search using Pydantic StrEnum.
    """
    HOUR = "qdr:h"
    DAY = "qdr:d"
    WEEK = "qdr:w"
    MONTH = "qdr:m"
    YEAR = "qdr:y"

async def google_general_search_async(search_query: str, 
                                        time_span: Optional[TimeSpan] = None, 
                                        web_domain: Optional[str] = None) -> Optional[dict]:
    """
    Perform a Google search using the Serper API.
    
    Args:
        search_query (str): The search query string.
        time_span (Optional[TimeSpan], optional): The time span. Defaults to None.
            - Allowed:
                - "qdr:h" (for hour)
                - "qdr:d" (for day)
                - "qdr:w" (for week)
                - "qdr:m" (for month)
                - "qdr:y" (for year)
        web_domain (Optional[str], optional): Search inside a web domain (e.g., web_domain="brainchip.com" -> searches only pages with this domain)
    Returns:
        Optional[dict]: The search results.
    """

    if web_domain and ("site:" not in search_query):
        if "site:" not in web_domain:
            web_domain = f"site:{web_domain}"
        search_query = f"{web_domain} {search_query}"

    if not search_query.strip():
        raise ValueError("Search query cannot be empty")
    if time_span and time_span not in ["qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"]:
        raise ValueError(f"Invalid time span value: {time_span}")

    num_results = 10

    payload = {
        "q": search_query,
        "num": num_results
    }
    
    if time_span:
        payload["tbs"] = time_span
        
    headers = {
        'X-API-KEY': config.SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(config.SERPER_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

async def google_scholar_search_async(search_query: str, num_pages: int = 1) -> dict:
    """Async version of google scholar search"""
    page = 1
    papers = []

    async with httpx.AsyncClient() as client:
        for _ in range(num_pages):
            payload = json.dumps({
                "q": search_query,
                "page": page
            })
            headers = {
                'X-API-KEY': config.SERPER_API_KEY,
                'Content-Type': 'application/json'
            }

            response = await client.post(
                config.SERPER_BASE_URL,
                headers=headers,
                data=payload
            )
            if response.status_code != 200:
                print(f"Scholar search failed with status {response.status_code}")
                return []
            json_response = response.json()
            papers.extend(json_response['organic'])
            page += 1

    return papers

async def google_news_search_async(search_query: str, num_pages: int = 1) -> dict:
    """Async version of google news search using Serper API
    
    Args:
        search_query (str): The search query to use
        num_pages (int, optional): Number of pages to fetch. Defaults to 1.
        
    Returns:
        dict: List of news articles found
    """
    page = 1
    news_articles = []

    async with httpx.AsyncClient() as client:
        for _ in range(num_pages):
            payload = json.dumps({
                "q": search_query,
                "page": page
            })
            headers = {
                'X-API-KEY': config.SERPER_API_KEY,
                'Content-Type': 'application/json'
            }

            response = await client.post(
                config.SERPER_BASE_URL,
                headers=headers,
                data=payload
            )
            if response.status_code != 200:
                print(f"News search failed with status {response.status_code}")
                return []
            json_response = response.json()
            news_articles.extend(json_response.get('news', []))
            page += 1

    return news_articles

class PerplexityResult(TypedDict):
    text_response: str
    citations: list[str]

async def perplexity_search_async(search_query: str) -> PerplexityResult | None:
    """Async version using AsyncOpenAI"""
    try:
        client = AsyncOpenAI(
            api_key=config.PERPLEXITY_API_KEY,
            base_url=config.PERPLEXITY_BASE_URL
        )

        response = await client.chat.completions.create(
            model="sonar-pro",
            messages=[{
                "role": "system",
                "content": "You are an artificial intelligence assistant that engages in helpful, detailed conversations.",
            }, {   
                "role": "user",
                "content": search_query,
            }],
        )

        message = response.choices[0].message.content + "\n\n"

        response = PerplexityResult(
            text_response = message.strip(),
            citations=response.citations
        )

        return response
    except Exception as e:
        print(f"Perplexity search failed: {e}")
        return None

def perplexity_sonar_reasoning(search_query: str) -> PerplexityResult | None:

    try:
        client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        )

        completion = client.chat.completions.create(
        model=config.OPENROUTER_PERPLEXITY_SONAR_REASONING,
        messages=[
            {
            "role": "user",
            "content": search_query
            }
        ])

        response = PerplexityResult(
                text_response = completion.choices[0].message.content.strip(),
                citations=completion.citations
            )

        return response
    except Exception as e:
        print(f"Perplexity search failed: {e}")
        return None

async def papers_with_code_search_async(query: str, items_per_page: int = 200) -> dict | None:
    """Async version of papers with code search"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://paperswithcode.com/api/v1/search/",
                params={"items_per_page": items_per_page, "q": query},
                headers={"accept": "application/json", "X-CSRFToken": config.PAPERS_WITH_CODE_CSRF_TOKEN}
            )
            response.raise_for_status()
            data = response.json()
            sorted_list = sorted(data['results'], key=lambda x: x['repository']['stars'], reverse=True)
            data['results'] = sorted_list
            return data
    except Exception as e:
        print(f"Request failed: {e}")
        return None

async def map_website_async(url: str, include_subdomains: bool = True) -> dict | None:
    """Map a website's content using FirecrawlApp.
    
    Args:
        url (str): The URL to map
        include_subdomains (bool, optional): Whether to include subdomains in the mapping. Defaults to True.
        
    Returns:
        dict | None: The mapping results or None if the operation fails
    """
    try:
        app = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)
        result = app.map_url(url, params={
            'includeSubdomains': include_subdomains
        })
        return result
    except Exception as e:
        print(f"Website mapping failed: {e}")
        return None

def scrape_website_firecrawl(url: str) -> dict | None:
    """Scrape a website's content using FirecrawlApp.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        dict | None: The scraping results or None if the operation fails
    """
    try:
        app = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)
        result = app.scrape_url(url, params={
            'formats': ['markdown']
        })
        return result
    except Exception as e:
        print(f"Website scraping failed: {e}")
        return None

def crawl_website_firecrawl(url: str, limit: int = 10) -> dict | None:
    """Crawl a website's content using FirecrawlApp.
    
    Args:
        url (str): The URL to crawl
        limit (int, optional): Maximum number of pages to crawl. Defaults to 10.
        
    Returns:
        dict | None: The crawling results or None if the operation fails
    """
    try:
        app = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)
        result = app.crawl_url(url, params={
            'limit': limit,
            'scrapeOptions': {
                'formats': ['markdown']
            }
        })
        return result
    except Exception as e:
        print(f"Website crawling failed: {e}")
        return None

async def crawl4ai_website_async(url_webpage: str) -> str:

    """
    Crawl a website using the crawl4ai library.
    
    Args:
        url_webpage (str): The URL of the webpage to crawl.
    
    Returns:
        str: The crawled content in markdown format.
    """
    md = MarkItDown()

    if is_pdf_url(url_webpage):
        result = md.convert(url_webpage)
        return result.text_content
        
    # Download the HTML content
    async with aiohttp.ClientSession() as session:
        async with session.get(url_webpage) as response:
            html_text = await response.text()

    # Extract text from the HTML to determine if it has useful content
    soup = BeautifulSoup(html_text, 'html.parser')
    text_content = soup.get_text(strip=True)

    # If the extracted text is insufficient, likely due to JavaScript rendering issues, use the crawler
    if len(text_content) < 100:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url_webpage)
            return result.markdown
    else:
        result = md.convert(url_webpage)
        return result.text_content

class ReasoningModelResponse(BaseModel):
    reasoning_content: Optional[str] = Field(description="The reasoning/thinking chain-of-thought output of the model.")
    final_answer: str = Field(description="The final response/answer of the model (after thinking).")

def deepseekR1_call(user_input: str) -> ReasoningModelResponse:
    """
    Call the DeepSeek Reasoner model to process user input.

    This function sends a user query to the DeepSeek Reasoner model and processes
    the streamed response, separating the reasoning content from the final answer.

    Args:
        user_input (str): The user's query or input to be processed by the model.

    Returns:
        DeepseekR1Response: An object containing the reasoning content and final answer.

    Raises:
        Exception: If there's an error in API communication or response processing.
    """

    deepseek_client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL
    )
    model = config.DEEPSEEK_R1

    deepseek_messages = []
    deepseek_messages.append({
        "role": "user", 
        "content": user_input
        })
    
    response = deepseek_client.chat.completions.create(
                model=model,
                #max_tokens=1,
                messages=deepseek_messages,
                stream=True
            )

    reasoning_content = ""
    final_content = ""

    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_piece = chunk.choices[0].delta.reasoning_content
            reasoning_content += reasoning_piece
        elif chunk.choices[0].delta.content:
            final_content += chunk.choices[0].delta.content

    response = ReasoningModelResponse(
        reasoning_content=reasoning_content, 
        final_answer=final_content)
    return response        

def openrouter_deepseekR1_call(user_input: str) -> ReasoningModelResponse:

    url = f"{config.OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.OPENROUTER_DEEPSEEK_R1,
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "include_reasoning": True
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    thinking_part = response.json()['choices'][0]['message']['reasoning']
    final_answer = response.json()['choices'][0]['message']['content']

    response = ReasoningModelResponse(
        reasoning_content=thinking_part, 
        final_answer=final_answer)
    return response 

def gemini_flash2_thinking_call(user_input: str) -> ReasoningModelResponse:
    
    # Only run this block for Gemini Developer API
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.FLASH2T_MODEL,
        contents=user_input,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            http_options=types.HttpOptions(api_version='v1alpha'),
        )
    )
    if len(response.candidates[0].content.parts) > 1:

        thinking_part = response.candidates[0].content.parts[0].text
        final_answer = response.candidates[0].content.parts[1].text

        response = ReasoningModelResponse(
            reasoning_content=thinking_part, 
            final_answer=final_answer)
    else:
        response = ReasoningModelResponse(
            reasoning_content=None, 
            final_answer=response.candidates[0].content.parts[0].text)
        
    return response 
