# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Critic agent for identifying and verifying statements using search tools."""

import os
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
#from google.adk.models import LlmResponse, Gemini
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm  #  Import LiteLlm for LM Studio
#from google.adk.tools import google_search
from google.genai import types
from duckduckgo_search import DDGS  # pip install duckduckgo-search

retry_options = types.HttpRetryOptions(initial_delay=1, attempts=6)

import sys
sys.path.append("../..")
from callback_logging import log_query_to_model, log_model_response

from . import prompt

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

def _render_reference(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    """Appends grounding references to the response."""
    del callback_context
    if (
        not llm_response.content or
        not llm_response.content.parts or
        not llm_response.grounding_metadata
    ):
        return llm_response
    references = []
    for chunk in llm_response.grounding_metadata.grounding_chunks or []:
        title, uri, text = '', '', ''
        if chunk.retrieved_context:
            title = chunk.retrieved_context.title
            uri = chunk.retrieved_context.uri
            text = chunk.retrieved_context.text
        elif chunk.web:
            title = chunk.web.title
            uri = chunk.web.uri
        parts = [s for s in (title, text) if s]
        if uri and parts:
            parts[0] = f'[{parts[0]}]({uri})'
        if parts:
            references.append('* ' + ': '.join(parts) + '\n')
    if references:
        reference_text = ''.join(['\n\nReference:\n\n'] + references)
        llm_response.content.parts.append(types.Part(text=reference_text))
    if all(part.text is not None for part in llm_response.content.parts):
        all_text = '\n'.join(part.text for part in llm_response.content.parts)
        llm_response.content.parts[0].text = all_text
        del llm_response.content.parts[1:]
    return llm_response

#modify critic agent to use LiteLlm and custom web search tool instead of google_search tool to support local models
critic_agent = Agent(
    #model=Gemini(model=os.getenv("MODEL"), retry_options=retry_options),
    model = LiteLlm(
        model="openai/gemma-3-4b",
        #provider="openai",
        api_base="http://127.0.0.1:1234/v1",
        api_key="not-needed"
    ),
    name='critic_agent',
    instruction=prompt.CRITIC_PROMPT,
    #tools=[google_search],
    tools=[web_search], # to support local models, we use a custom web search function instead of the google_search tool
    before_model_callback=log_query_to_model,
    after_model_callback=_render_reference,
    
)
