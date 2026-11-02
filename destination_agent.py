from pydantic import BaseModel, Field
from typing import List
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from state import destination_agent_prompt, TravelState

load_dotenv()

class Attraction(BaseModel):
    name: str
    location: str
    category: str
    description: str
    duration_hours: float
    priority: str
    source_url: str

class Region(BaseModel):
    name: str
    recommended_days: int
    attractions: List[Attraction]

class DestinationResearch(BaseModel):
    destination: str
    overview: str
    regions: List[Region]
    seasonal_considerations: List[str]
    travel_restrictions: List[str]
    important_notes: List[str]


tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    topic="general",
    include_answer=False,
    include_raw_content=True
)

model = init_chat_model("google_genai:gemini-2.5-flash")

destination_agent = create_react_agent(
    model=model,
    tools=[tavily_tool],
    #agent_name="Destination Research Agent",
    response_format=DestinationResearch,
    prompt=destination_agent_prompt
)

def destination_node(state: TravelState):

    task = state["agent_tasks"]["destination"]
    print(f"[destination] task received: {task[:100]}...")

    response = None
    searches = 0
    for chunk in destination_agent.stream({
        "messages": [
            {
                "role": "user",
                "content": task
            }
        ]
    }, stream_mode="values"):
        response = chunk
        last = chunk["messages"][-1]
        for call in getattr(last, "tool_calls", None) or []:
            searches += 1
            print(f"[destination] search #{searches}: {call['args'].get('query')}")

    result = response["structured_response"]
    print(f"[destination] done: {len(result.regions)} regions, "
          f"{sum(len(r.attractions) for r in result.regions)} attractions, {searches} searches")

    return {
        "agent_results": {
            **state["agent_results"],
            "destination": {
                "status": "completed",
                "data": response[
                    "structured_response"
                ].model_dump()
            }
        }
    }




