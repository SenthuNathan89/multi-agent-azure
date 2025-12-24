import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from tools.toolkit import summarize_text
from common.common import MultiAgentState
    
class SummaryAgent:
    "Agent specialized in text summarization and analysis."
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [summarize_text]
        self.tool_node = ToolNode(self.tools)
        self.name = "Summary Agent"
        
        self.system_prompt = """You are a specialized Summary Agent. Your expertise is in:
- Summarizing long documents and texts
- Extracting key points
- Creating concise overviews
- Analyzing and condensing information

Use the summarize_text tool for long content. Provide clear, structured summaries.
"""
    
    def process(self, state: MultiAgentState):
        # Process summarization queries
        messages = state["messages"]
        system_msg = SystemMessage(content=self.system_prompt)
        full_messages = [system_msg] + list(messages)
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    def should_use_tools(self, state: MultiAgentState):
        # Check if tools are needed
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "complete"