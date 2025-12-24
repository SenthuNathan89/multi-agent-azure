import operator
import os
from typing import TypedDict, Annotated, Sequence, Dict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import BaseMessage
from tools.ragSearch import AzureSearchVector
from dotenv import load_dotenv  

load_dotenv()

class MultiAgentState(TypedDict):
    "State shared across all agents in the system."
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str  # Which agent should handle this next
    final_response: str  # Final answer to return to user
    task_context: dict  # Additional context passed between agents such as which tools are available on each agent

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature = 0.7
)

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint = os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
    api_key = os.getenv("AZURE_EMBEDDINGS_API_KEY"),
    azure_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
    api_version = os.getenv("AZURE_EMBEDDINGS_API_VERSION")
)      
# Initialize vector store
vector_store = AzureSearchVector(
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"),
    key = os.getenv("AZURE_SEARCH_KEY"),
    embeddings=embeddings,
    vector_field="text_vector",
    text_field="chunk",  
)