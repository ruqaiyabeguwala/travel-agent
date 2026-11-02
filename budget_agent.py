from pydantic import BaseModel, Field
from typing import Literal
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from state import budget_agent_prompt, TravelState
import json

load_dotenv()


class CostItem(BaseModel):
    category: Literal[
        "transport", "accommodation", "food", "permits_and_fees",
        "activities", "miscellaneous"
    ]
    description: str
    estimated_cost: float | None = None
    currency: str = "INR"
    price_status: Literal[
        "estimate", "live_verified", "not_found"
    ]
    source_urls: list[str] = Field(default_factory=list)


class BudgetResearch(BaseModel):
    overview: str = Field(description="2-3 sentence summary of the total cost, budget fit and main cost drivers")
    currency: str = "INR"
    total_estimated_cost: float | None = None
    cost_per_person: float | None = None
    cost_breakdown: list[CostItem]
    user_budget: float | None = None
    within_budget: bool | None = None
    budget_difference: float | None = Field(
        default=None,
        description="user_budget minus total_estimated_cost; negative means over budget"
    )
    saving_suggestions: list[str] = Field(default_factory=list)
    important_notes: list[str] = Field(default_factory=list)


tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    topic="general",
    include_answer=False,
    include_raw_content=True
)

budget_model = init_chat_model("google_genai:gemini-2.5-flash")

budget_agent = create_react_agent(
    model=budget_model,
    tools=[tavily_tool],
    prompt=budget_agent_prompt,
    response_format=BudgetResearch
)


def budget_node(state: TravelState):

    task = state.get("agent_tasks", {}).get(
        "budget",
        "Estimate the total trip cost and check it against the budget."
    )

    agent_results = state.get("agent_results", {})
    context = {
        "travel_request": state["travel_request"],
        "destination_research": agent_results.get("destination", {}),
        "transport_research": agent_results.get("transport", {}),
        "hotel_research": agent_results.get("hotel", {}),
        "task": task
    }
    print(f"[budget] task received: {task[:100]}...")
    print(f"[budget] using research from: {[a for a in ('destination', 'transport', 'hotel') if agent_results.get(a)] or 'none'}")

    response = None
    searches = 0
    for chunk in budget_agent.stream({
        "messages": [
            HumanMessage(content=json.dumps(context))
        ]
    }, stream_mode="values"):
        response = chunk
        last = chunk["messages"][-1]
        for call in getattr(last, "tool_calls", None) or []:
            searches += 1
            print(f"[budget] search #{searches}: {call['args'].get('query')}")

    result = response["structured_response"]
    total = f"{result.currency} {result.total_estimated_cost:,.0f}" if result.total_estimated_cost is not None else "n/a"
    print(f"[budget] done: total {total}, within budget: {result.within_budget}, "
          f"{len(result.cost_breakdown)} cost items, {searches} searches")

    results = dict(agent_results)

    results["budget"] = {
        "status": "completed",
        "data": result.model_dump()
    }

    return {
        "agent_results": results
    }
