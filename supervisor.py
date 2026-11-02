from pydantic import BaseModel, Field
from state import SupervisorDecision, TravelState, system_prompt, itinerary_prompt
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import json

load_dotenv()

model = init_chat_model("google_genai:gemini-3.6-flash")

supervised_model = model.with_structured_output(SupervisorDecision)

def supervisor_agent(state: TravelState):
    request = state["travel_request"]
    print(f"[supervisor] planning {request.get('origin')} -> {request.get('destination')}, "
          f"{request.get('duration_days')} days; existing results: {list(state['agent_results']) or 'none'}")
    prompt = f"""{system_prompt}
    """
    response = supervised_model.invoke([{
        "role": "system",
        "content": prompt,
    },
    {
        "role": "user",
        "content": f"""
        Travel Request:{state['travel_request']}

        Existing Agent Results:
        {state['agent_results']}

        Decide the next action.
        """
    }]
    )
    print(f"[supervisor] action={response.action}, agents={response.agents_to_call}, reason: {response.reason}")
    if response.clarification_question:
        print(f"[supervisor] asks user: {response.clarification_question}")
    return response


class ItineraryDay(BaseModel):
    day: int
    title: str
    region: str
    activities: list[str]
    accommodation: str | None = None
    transport: str | None = None
    estimated_cost: float | None = None


class Itinerary(BaseModel):
    overview: str = Field(description="2-3 sentence summary of the trip plan and its estimated cost")
    days: list[ItineraryDay]
    currency: str = "INR"
    total_estimated_cost: float | None = None
    important_notes: list[str] = Field(default_factory=list)


itinerary_model = model.with_structured_output(Itinerary)


def itinerary_node(state: TravelState):
    agent_results = state["agent_results"]
    print(f"[itinerary] supervisor received full results from: {list(agent_results)}")

    # the entire data of every agent is passed, nothing is trimmed
    response = itinerary_model.invoke([
        {"role": "system", "content": itinerary_prompt},
        {"role": "user", "content": f"""
Travel Request:
{json.dumps(state['travel_request'], default=str)}

Complete Agent Results:
{json.dumps(agent_results, default=str)}

Create the final day-wise itinerary.
"""}
    ])
    print(f"[itinerary] done: {len(response.days)} days planned, total cost: {response.total_estimated_cost}")

    return {
        "agent_results": {
            **agent_results,
            "itinerary": {
                "status": "completed",
                "data": response.model_dump()
            }
        }
    }
