from langgraph.graph import StateGraph, START, END
from state import TravelState
from supervisor import supervisor_agent, itinerary_node
from destination_agent import destination_node
from transport_agent import transport_node
from hotel_agent import hotel_node
from budget_agent import budget_node
from langgraph.checkpoint.memory import InMemorySaver

builder = StateGraph(TravelState)

builder.add_node(
    "supervisor",
    supervisor_agent
)

builder.add_node(
    "destination",
    destination_node
)

builder.add_node(
    "transport",
    transport_node
)

builder.add_node(
    "hotel",
    hotel_node
)

builder.add_edge(START, "supervisor")
builder.add_node(
    "budget",
    budget_node
)

builder.add_node(
    "itinerary",
    itinerary_node
)

builder.add_edge("supervisor", "destination")
builder.add_edge("destination", "transport")
builder.add_edge("transport", "hotel")
builder.add_edge("hotel", "budget")
builder.add_edge("budget", "itinerary")
builder.add_edge("itinerary", END)


checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

initial_state = {
    "travel_request": {
        "origin": "Indore",
        "destination": "Bhutan",
        "duration_days": 7,
        "travellers": 2,
        "budget": 150000,
        "currency": "INR",
        "interests": [
            "mountains",
            "nature",
            "adventure"
        ],
        "travel_dates": None
    },

    "agent_tasks": {},

    "agent_results": {},

    "supervisor_decision": {},

    "workflow_status": "started",

    "clarification_question": ""
}

if __name__ == "__main__":
    print("[main] starting travel planning graph")
    result = graph.invoke(initial_state)

    print(f"[main] finished: agents completed = {list(result['agent_results'])}")
    print(result)