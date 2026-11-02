from pydantic import BaseModel, Field
from typing import Literal
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from state import hotel_agent_prompt, TravelState
import json
from langchain_core.messages import HumanMessage

load_dotenv()


tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    topic="general",
    include_answer=False,
    include_raw_content=True
)

class Hotel(BaseModel):
    name: str
    location: str
    property_type: str

    description: str
    rating: float | None = None
    rating_source: str | None = None

    price_per_night: float | None = None
    currency: str = "INR"

    price_status: Literal[
        "estimate",
        "live_verified",
        "not_found"
    ]

    amenities: list[str] = Field(default_factory=list)

    nearby_attractions: list[str] = Field(
        default_factory=list
    )

    source_urls: list[str] = Field(default_factory=list)


class HotelRegion(BaseModel):
    region: str
    hotels: list[Hotel]


class HotelResearch(BaseModel):
    overview: str = Field(description="2-3 sentence summary of the recommended stay options, price range and best areas to stay")
    destination: str
    number_of_nights: int
    regions: list[HotelRegion]
    estimated_accommodation_cost: float | None = None
    currency: str = "INR"
    important_notes: list[str] = Field(default_factory=list)


hotel_model = init_chat_model("google_genai:gemini-2.5-flash")

hotel_agent = create_react_agent(
    model=hotel_model,
    tools=[tavily_tool],
    prompt=hotel_agent_prompt,
    response_format=HotelResearch
)


def hotel_node(state: TravelState):

    context = {
        "travel_request": state["travel_request"],
        "destination_research": state["agent_results"].get(
            "destination", {}
        ),
        "transport_research": state["agent_results"].get(
            "transport", {}
        ),
        "task": state["agent_tasks"].get(
            "hotel",
            "Research suitable accommodation."
        )
    }

    print(f"[hotel] task received: {context['task'][:100]}...")
    print(f"[hotel] using research from: {[a for a in ('destination', 'transport') if state['agent_results'].get(a)] or 'none'}")

    response = None
    searches = 0
    for chunk in hotel_agent.stream({
        "messages": [
            HumanMessage(content=json.dumps(context))
        ]
    }, stream_mode="values"):
        response = chunk
        last = chunk["messages"][-1]
        for call in getattr(last, "tool_calls", None) or []:
            searches += 1
            print(f"[hotel] search #{searches}: {call['args'].get('query')}")

    result = response["structured_response"]
    print(f"[hotel] done: {sum(len(r.hotels) for r in result.regions)} hotels in "
          f"{len(result.regions)} regions, {searches} searches")

    results = dict(state.get("agent_results", {}))

    results["hotel"] = {
        "status": "completed",
        "data": response[
            "structured_response"
        ].model_dump()
    }

    return {
        "agent_results": results
    }



