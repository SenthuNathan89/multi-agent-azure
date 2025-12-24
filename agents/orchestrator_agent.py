import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from common.common import MultiAgentState

class Orchestrator:
    "Orchestrator that routes requests to appropriate specialized agents."
    def __init__(self, llm):
        self.llm = llm
        self.name = "Orchestrator"

    def route(self, state: MultiAgentState):
        # Determine which agent should handle the request
        messages = state["messages"]
        # Get the latest user message
        user_message = messages[-1].content if messages else ""
        # Create routing prompt
        routing_system_prompt = """Analyze this query and decide routing.

If query requires MULTIPLE steps/agents, respond: PLANNER_AGENT
If query is simple (one agent), respond with: MATH_AGENT, RESEARCH_AGENT, SUMMARY_AGENT, or GENERAL_AGENT

Examples:
"Research AI trends and calculate growth" -> PLANNER_AGENT (needs research THEN math)
"Find climate info and summarize it" -> PLANNER_AGENT (needs research THEN summary)
"What is 50 * 89?" -> MATH_AGENT (simple, one agent)
"Search for Python tutorials" -> RESEARCH_AGENT (simple, one agent)
"Hello" -> BASE_AGENT (simple)

Query: {query}
"""
        # Simple message format
        routing_messages = [
            SystemMessage(content=routing_system_prompt.format(query=user_message)),
            # HumanMessage(content=user_message)
        ]
        # Ask LLM to route
        response = self.llm.invoke(routing_messages)
        # Parse the response to get agent name
        agent_name = response.content.strip().upper()
        # Validate agent name
        valid_agents = ["PLANNER_AGENT", "COORDINATOR_AGENT", "MATH_AGENT", "RESEARCH_AGENT", "SUMMARY_AGENT", "BASE_AGENT"]
        if agent_name not in valid_agents:
            agent_name = "BASE_AGENT"  # Default fallback agent
        print(f"\n Orchestrator routing to: {agent_name}")
        
        return {"next_agent": agent_name.lower()}

