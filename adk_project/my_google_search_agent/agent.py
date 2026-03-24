import os
from dotenv import load_dotenv

from google.adk import Agent
#from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm  #  Import LiteLlm for LM Studio
from google.genai import types
#from google.adk.tools import google_search  # The Google Search tool not supporterd for local models
from duckduckgo_search import DDGS  # pip install duckduckgo-search



load_dotenv()
model_name = os.getenv("MODEL")

# Retry options help avoid the occasional error from popular models
# receiving too many requests at once.
retry_options = types.HttpRetryOptions(initial_delay=1, attempts=6)

import sys
sys.path.append("..")
from callback_logging import log_query_to_model, log_model_response

# Custom search tool — works with any LM Studio model
def web_search(query: str) -> str:
    """Search the web for current information on a given query.
    
    Args:
        query: The search query string.
    
    Returns:
        A string of search results with titles and snippets.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    
    if not results:
        return "No results found."
    
    return "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}"
        for r in results
    )

root_agent = Agent(
    # name: A unique name for the agent.
    name="google_search_agent",
    # description: A short description of the agent's purpose, so
    # other agents in a multi-agent system know when to call it.
    description="Answer questions using Google Search.",
    # model: The LLM model that the agent will use:
    #model=Gemini(model=model_name, retry_options=retry_options),
    model=LiteLlm(
            model="openai/gemma-3-4b",#throws ValueError: Google search tool is not supported for model openai/gemma-3-4b
            #provider="openai",
            api_base="http://127.0.0.1:1234/v1",  # LM Studio local server
            api_key="not-needed"
        ),
    # instruction: Instructions (or the prompt) for the agent.
    instruction="You are an expert researcher. You stick to the facts.",
    # callbacks: Allow for you to run functions at certain points in
    # the agent's execution cycle. In this example, you will log the
    # request to the agent and its response.
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    
    # tools: functions to enhance the model's capabilities.
    # Add the google_search tool below.
    #tools=[google_search]
    tools=[web_search] # to support local models, we use a custom web search function instead of the google_search tool
    

)
