import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from tools.toolkit import calculate
from common.common import MultiAgentState

class MathAgent:
    "Agent specialized in mathematical calculations."
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [calculate]
        self.tool_node = ToolNode(self.tools)
        self.name = "Math Agent"
        self.system_prompt = """You are a specialized Math Agent. Your expertise is in:
- Performing mathematical calculations
- Solving equations
- Statistical analysis
- Mathematical reasoning
Use the calculate tool to perform computations. Be precise and explain your work.
"""
    def process(self, state: MultiAgentState):
        # Process mathematical queries
        messages = state["messages"]
        # Add system prompt
        system_msg = SystemMessage(content=self.system_prompt)
        full_messages = [system_msg] + list(messages)
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}
    
    def should_use_tools(self, state: MultiAgentState):
        # Check if tools are needed
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "complete"