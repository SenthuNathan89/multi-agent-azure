import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from common.common import MultiAgentState

# DO NOT REMOVE PLANNER AGENT, ORCHESTRATOR AGENT OR COORDINATOR AGENT IN THE TEMPLATE AS THEY ARE CRUCIAL IN THE WORKFLOW. THE OTHERS CAN BE EDITED OR REMOVED.

class PlannerAgent:
    "Agent that breaks down complex tasks into sequential steps."
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Planner Agent"
        self.system_prompt = """You are a Planner Agent responsible for task decomposition.

When given a complex request, analyze it and create a sequential execution plan.
Output your plan in this EXACT format:
STEP 1: [task description] -> [agent_name]
STEP 2: [task description] -> [agent_name]
STEP 3: [task description] -> [agent_name]

Available agents:
- MATH_AGENT: calculations, equations, statistics
- RESEARCH_AGENT: searching, finding information
- SUMMARY_AGENT: summarizing, condensing text
- GENERAL_AGENT: general conversation

Examples:
Query: "Research AI market trends and calculate the growth rate"
STEP 1: Research current AI market data -> RESEARCH_AGENT
STEP 2: Calculate growth rate from the data -> MATH_AGENT

Query: "Find information about climate change and summarize it"
STEP 1: Search for climate change information -> RESEARCH_AGENT
STEP 2: Summarize the findings -> SUMMARY_AGENT

Query: "What is 50 * 89?"
STEP 1: Calculate 50 * 89 -> MATH_AGENT

If the task is simple (only needs one agent), output just one step.
"""
    
    def process(self, state: MultiAgentState):
        # Analyze query and create execution plan.
        messages = state["messages"]
        
        system_msg = SystemMessage(content=self.system_prompt)
        user_query = messages[-1].content
        
        planning_messages = [system_msg, HumanMessage(content=user_query)]
        response = self.llm.invoke(planning_messages)
    
        # Parse the plan
        plan = self._parse_plan(response.content)
        
        # Store plan in task_context
        state["task_context"]["plan"] = plan
        state["task_context"]["current_step"] = 0
        state["task_context"]["plan_results"] = []
        
        # Route to first agent
        if plan:
            first_agent = plan[0]["agent"]
            print(f"\n Plan created with {len(plan)} steps")
            for i, step in enumerate(plan, 1):
                print(f"   Step {i}: {step['task']} -> {step['agent']}")
            return {"next_agent": first_agent, "task_context": state["task_context"]}
        
        return {"next_agent": "base_agent"}
    
    def _parse_plan(self, plan_text): # Instead of calling and deciding which tools we need for the agent, the planner agent should have a function on how parse the output plan from the LLM
        # Parse the plan into structured steps.
        steps = []
        for line in plan_text.split('\n'):
            if line.strip().startswith('STEP'):
                try:
                    # Extract task and agent
                    parts = line.split('->')
                    if len(parts) == 2:
                        task = parts[0].split(':', 1)[1].strip()
                        agent = parts[1].strip().lower()
                        steps.append({"task": task, "agent": agent})
                except:
                    continue
        return steps