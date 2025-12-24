import os
import operator
import sqlite3
import phoenix as px
from dotenv import load_dotenv  
from phoenix.otel import register       

from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from memory.memory import get_session_history, clear_session_history, list_sessions
from tools.toolkit import calculate, summarize_text, search_knowledge_base, web_search
from tools.ragSearch import AzureSearchVector

from agents.base_agent import BaseAgent
from agents.math_agent import MathAgent
from agents.research_agent import ResearchAgent
from agents.summary_agent import SummaryAgent
from agents.orchestrator_agent import Orchestrator
from agents.planner_agent import PlannerAgent
from agents.coordinator_agent import CoordinatorAgent

#Azure Components Initialization - LLM, Embeddings, Vector Store, Memory---------------------------------------------------------------------------
from common.common import MultiAgentState, llm, embeddings, vector_store # Multi-agent defnition

# Monitoring Initialization---------------------------------------------------------------------------------------------------------------
# os.environ["PHOENIX_WORKING_DIR"] = "./phoenix_data"
# px.launch_app()
# tracer_provider = register(
#   project_name="Azure_LLM_Agent",
#   endpoint="http://localhost:6006/v1/traces",
#   auto_instrument=True
# )

# Load environment variables----------------------------------------------------------------------------------------------------------------------
load_dotenv()
SQLITE_DB_PATH ="chat_history.db"

# SQLite-based chat history management-----------------------------------------------------------------------------------------------------------------
chat_history = {}

# Multi Agent System Initialization and Graph Workflow ------------------------------------------------------------------------------------------------
class MultiAgentSystem:
    "Main multi-agent system coordinating all agents."
    
    def __init__(self, llm, embeddings, vector_store):
        self.llm = llm
        # Initialize all agents - Everytime you create an agent; add it here
        self.orchestrator = Orchestrator(llm) # Can cahnge the llm models per agent here but we are calling only one now
        self.math_agent = MathAgent(llm)
        self.research_agent = ResearchAgent(llm)
        self.summary_agent = SummaryAgent(llm)
        self.base_agent = BaseAgent(llm)
        self.planner_agent = PlannerAgent(llm)
        self.coordinator_agent = CoordinatorAgent(llm)

        # Build the workflow
        self.app = self._build_workflow()
    
    def _build_workflow(self):
        "Build the multi-agent LangGraph workflow with a multi agent execution for complex prompts and tasks"
        workflow = StateGraph(MultiAgentState)
        # Add all the nodes; each time an agent is created add the agent node.tools if tools are present and agent.process for process flow
        workflow.add_node("orchestrator", self.orchestrator.route)
        workflow.add_node("planner_agent", self.planner_agent.process)       
        workflow.add_node("coordinator_agent", self.coordinator_agent.process)
        workflow.add_node("math_agent", self.math_agent.process)
        workflow.add_node("math_tools", self.math_agent.tool_node) # Tool node necessary for the agent that has to perform tool calls
        workflow.add_node("research_agent", self.research_agent.process)
        workflow.add_node("research_tools", self.research_agent.tool_node)# Tool node necessary for the agent that has to perform tool calls
        workflow.add_node("summary_agent", self.summary_agent.process)
        workflow.add_node("summary_tools", self.summary_agent.tool_node)# Tool node necessary for the agent that has to perform tool calls
        workflow.add_node("general_agent", self.base_agent.process)
        # Set entry point
        workflow.set_entry_point("orchestrator")

        # Orchestrator routes to agents
        workflow.add_conditional_edges(
            "orchestrator",
            lambda x: x["next_agent"],
            {
                "planner_agent": "planner_agent",              
                "math_agent": "math_agent",         
                "research_agent": "research_agent",
                "summary_agent": "summary_agent",
                "general_agent": "general_agent"
            }
        )
        workflow.add_conditional_edges(
            "planner_agent",
            lambda x: x["next_agent"],
            {
                "math_agent": "math_agent",
                "research_agent": "research_agent",
                "summary_agent": "summary_agent",
                "general_agent": "general_agent"
            }
        )
        # Add all edges based on the condition of each agent calling their respective tools
        # Math agent flow
        workflow.add_conditional_edges(
            "math_agent",
            self.math_agent.should_use_tools,
            {
                "tools": "math_tools",
                "complete": "coordinator_agent"
            }
        )
        workflow.add_edge("math_tools", "math_agent")
        # Research agent flow
        workflow.add_conditional_edges(
            "research_agent",
            self.research_agent.should_use_tools,
            {
                "tools": "research_tools",
                "complete": "coordinator_agent"
            }
        )
        workflow.add_edge("research_tools", "research_agent")
        # Summary agent flow
        workflow.add_conditional_edges(
            "summary_agent",
            self.summary_agent.should_use_tools,
            {
                "tools": "summary_tools",
                "complete": "coordinator_agent"
            }
        )
        workflow.add_edge("summary_tools", "summary_agent")
        workflow.add_edge("general_agent", "coordinator_agent")

        workflow.add_conditional_edges(
            "coordinator_agent",
            lambda x: x.get("next_agent", "end"),
            {
                "math_agent": "math_agent",
                "research_agent": "research_agent",
                "summary_agent": "summary_agent",
                "general_agent": "general_agent",
                "end": END
            }
        )
        
        return workflow.compile()

# INITIALIZING MULTI-AGENT SYSTEM ----------------------------------------------------------------------------------------------------------------------------
multi_agent_system = MultiAgentSystem(llm, embeddings, vector_store) 

def run_multi_agent(user_input: str, session_id: str = "defaultUser"):
        "Run the multi-agent system with memory."
        # Load previous messages
        chat_history = get_session_history(session_id)
        previous_messages = chat_history.messages
        initial_state = {
            "messages": previous_messages + [HumanMessage(content=user_input)],
            "next_agent": "",
            "final_response": "",
            "task_context": {"original_query": user_input} # To prevent previous messages from the chat history coming as the original query
        }
        # Run the compiled workflow through the multi-agent system
        result = multi_agent_system.app.invoke(initial_state)
        # Save to memory
        chat_history.add_user_message(user_input)
        # Get the final AI message
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            chat_history.add_ai_message(final_message.content)
            return final_message.content
        
        return str(final_message)

# ====================================================================================================================================================================================================
# INTERACTIVE MULTI AGENT SYSTEM CLI INTERFACE
# ====================================================================================================================================================================================================

def interactive_cli():
    "Main interactive CLI loop."
    print("\n" + "="*100)
    print("   INTERACTIVE AGENT CLI")
    print("="*100)
    print("\nCommands:")
    print("  - Type your query and press Enter to talk to the agent")
    print("  - 'status' - Show agent status")
    print("  - 'clear' - Clear current session history")
    print("  - 'sessions' - List all available sessions")
    print("  - 'session <name>' - Switch to a different session")
    print("\nAvailable Agents:")
    print("  - Math Agent - Perform math calculations")
    print("  - Summary Agent - Summarize long text")
    print("  - Research Agent - Search the knowledge base and web search via Tavily")
    print("  - Base Agent - Chat")
    print("\n" + "="*100 + "\n")

    print("Enter your username to start:")
    username = input(f"").strip()
    current_session = username
    print(f"Starting session: {current_session}")

    while True:
        try:
            # Get user input 
            user_input = input(f"\n[{current_session}] You: ").strip()
            # Handle empty input
            if not user_input:
                continue    
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye! Your are on your own now!\n")
                break          
            elif user_input.lower() == 'clear':
                if clear_session_history(current_session):
                    print(f"\n Cleared history for session '{current_session}'\n")
                else:
                    print(f"\n No history to clear for session '{current_session}'\n")
                continue          
            elif user_input.lower().startswith('session '):
                new_session = user_input.split(' ', 1)[1].strip()
                if new_session:
                    current_session = new_session
                    print(f"\n Switched to session: {current_session}\n")
                else:
                    print("\n Please provide a session name\n")
                continue
            elif user_input.lower() == 'sessions':
                result = get_session_history()
                continue
            
            # Run the agent
            print(f"\n[{current_session}] Agent: ", end="", flush=True)
            response = run_multi_agent(user_input, session_id=current_session)
            print(response)
        
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye! Stupid of me or beyond my control!\n")
            break
        
        except Exception as e:
            print(f"\n Error: {e}\n")

if __name__ == "__main__":
    interactive_cli()

