import operator
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from common.common import MultiAgentState

class CoordinatorAgent:
    "Coordinates multi-step agent execution."
    def __init__(self, llm):
        self.llm = llm
        self.name = "Coordinator Agent"
    
    def process(self, state: MultiAgentState):
        # Check if more steps needed or give final output
        task_context = state.get("task_context", {})
        plan = task_context.get("plan", [])
        current_step = task_context.get("current_step", 0)
        plan_results = task_context.get("plan_results", [])
        
        # Store result from previous agent
        if len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                plan_results.append({
                    "step": current_step,
                    "result": last_message.content
                })
        
        # Move to next step
        current_step += 1
        
        # Check if more steps remaining
        if current_step < len(plan):
            next_agent = plan[current_step]["agent"]
            print(f"\n Moving to step {current_step + 1}/{len(plan)}")
            
            # Update context with intermediate results
            task_context["current_step"] = current_step
            task_context["plan_results"] = plan_results
            
            return {
                "next_agent": next_agent,
                "task_context": task_context
            }
        
        # All steps complete - synthesize final answer
        print(f"\n All steps complete. Synthesizing final answer...")
        return self._synthesize_results(state, plan_results)
    
    def _synthesize_results(self, state, plan_results):
        # Combine results from all agents into final response.
        task_context = state.get("task_context", {}) # To prevent precvious convo messages comingin as a query
        original_query = task_context.get("original_query", "Unknown query")

        synthesis_prompt = f"""You are a Coordinator synthesizing results from multiple agents.
                            Original query: {original_query}
                            Results from agents:
                            """
        for result in plan_results:
            synthesis_prompt += f"\nStep {result['step'] + 1}: {result['result']}\n"
        synthesis_prompt += "\nProvide a comprehensive final answer that combines all the information."
        
        messages = [
            SystemMessage(content="You are a coordinator combining results from multiple agents."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "messages": [response],
            "final_response": response.content,
            "next_agent": "end"  # To prevent looping from coordinator agent
        }