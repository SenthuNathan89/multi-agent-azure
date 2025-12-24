import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from common.common import MultiAgentState

class BaseAgent:
    "Agent for general conversation and tasks not requiring specialized tools." # Needed as per langchain tempalte defnition if you dont include decription
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Base Agent"
        self.system_prompt = """You are a helpful General Assistant. You handle:
- General conversation
- Questions that don't require specialized tools
- Coordinating with other agents if needed
- Providing helpful, conversational responses
Be friendly, helpful, and concise.
"""
    
    def process(self, state: MultiAgentState):
        # Process general queries
        messages = state["messages"]
        system_msg = SystemMessage(content=self.system_prompt)
        full_messages = [system_msg] + list(messages)
        response = self.llm.invoke(full_messages)
        
        return {"messages": [response]}