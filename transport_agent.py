from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from state import transport_agent_prompt, TravelState
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
import json

transport_model = init_chat_model("google_genai:gemini-2.5-flash")

class TransportOption(BaseModel):
    mode: Literal[
        "flight", "train", "bus", "car", "taxi", "other"
    ]
    origin: str
    destination: str
    operator: str | None = None
    duration: str
    estimated_cost: float | None = None
    currency: str = "INR"
    price_status: Literal[
        "estimate", "live_verified", "not_found"
    ]
    source_urls: list[str] = Field(default_factory=list)


class LocalRoute(BaseModel):
    origin: str
    destination: str
    distance_km: float | None = None
    estimated_duration: str
    recommended_transport: str
    estimated_cost: float | None = None
    currency: str = "INR"
    source_urls: list[str] = Field(default_factory=list)


class TransportResearch(BaseModel):
    overview: str = Field(description="2-3 sentence summary of the best ways to get there and around")
    origin: str
    destination: str
    intercity_options: list[TransportOption]
    local_routes: list[LocalRoute]
    entry_points: list[str]
    transport_rules: list[str]
    important_notes: list[str]


tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    topic="general",
    include_answer=False,
    include_raw_content=True
)

transport_agent = create_react_agent(
    model=transport_model,
    tools=[tavily_tool],
    prompt=transport_agent_prompt,
    response_format=TransportResearch
)

def transport_node(state: TravelState):

    task = state.get("agent_tasks", {}).get(
        "transport",
        "Research intercity and local transportation."
    )

    research_context = {
        "travel_request": state["travel_request"],
        "destination_research": state.get(
            "agent_results", {}
        ).get("destination", {}),
        "task": task
    }

    print(f"[transport] task received: {task[:100]}...")
    print(f"[transport] using destination research: {'yes' if research_context['destination_research'] else 'no'}")

    response = None
    searches = 0
    for chunk in transport_agent.stream({
        "messages": [
            HumanMessage(
                content=json.dumps(research_context)
            )
        ]
    }, stream_mode="values"):
        response = chunk
        last = chunk["messages"][-1]
        for call in getattr(last, "tool_calls", None) or []:
            searches += 1
            print(f"[transport] search #{searches}: {call['args'].get('query')}")

    result = response["structured_response"]
    print(f"[transport] done: {len(result.intercity_options)} intercity options, "
          f"{len(result.local_routes)} local routes, {searches} searches")

    results = dict(state.get("agent_results", {}))

    results["transport"] = {
        "status": "completed",
        "data": response[
            "structured_response"
        ].model_dump()
    }

    return {
        "agent_results": results
    }