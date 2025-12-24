import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from tools.toolkit import search_knowledge_base, web_search
from common.common import MultiAgentState
    
class ResearchAgent:
    "Agent specialized in knowledge retrieval and research."
    def __init__(self, llm):
        self.llm = llm
        self.tools = [search_knowledge_base, web_search]
        self.tool_node = ToolNode(self.tools)
        self.name = "Research Agent"
        
        self.system_prompt = """You are a specialized Research Agent. Your expertise is in:
- Searching knowledge bases for relevant information
- Web searching for current information
- Synthesizing information from multiple sources
- Fact-checking and verification

Use search_knowledge_base for internal documents and web_search for current information.
Provide comprehensive, well-sourced answers.
"""
    
    def process(self, state: MultiAgentState):
        # Process research queries via RAG or web
        messages = state["messages"]
        system_msg = SystemMessage(content=self.system_prompt)
        full_messages = [system_msg] + list(messages)
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    def should_use_tools(self, state: MultiAgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "complete"