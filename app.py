import uuid
import streamlit as st
from pydantic import BaseModel
from typing import Optional
from langchain.chat_models import init_chat_model

from main import graph

st.set_page_config(page_title="Travel Planner", page_icon="🧭")
st.title("🧭 Travel Planner")

AGENT_LABELS = {
    "supervisor": "Supervisor – deciding what to research",
    "destination": "Destination Agent – searching the web",
    "transport": "Transport Agent – researching routes",
    "hotel": "Hotel Agent – researching accommodations",
    "budget": "Budget Agent – pricing the trip",
    "itinerary": "Supervisor – building the day-wise itinerary from all results",
}


class ExtractedRequest(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    duration_days: Optional[int] = None
    travellers: Optional[int] = None
    budget: Optional[int] = None
    currency: str = "INR"
    interests: list[str] = []
    travel_dates: Optional[str] = None


extractor = init_chat_model("google_genai:gemini-2.5-flash").with_structured_output(ExtractedRequest)


def render_destination(data: dict):
    st.write(data["overview"])
    for region in data["regions"]:
        with st.expander(f"📍 {region['name']} – {region['recommended_days']} day(s)"):
            for a in region["attractions"]:
                st.markdown(
                    f"**{a['name']}** ({a['category']}, ~{a['duration_hours']}h, {a['priority']})  \n"
                    f"{a['description']}  \n[source]({a['source_url']})"
                )
    for title, key in [("Seasonal considerations", "seasonal_considerations"),
                       ("Travel restrictions", "travel_restrictions"),
                       ("Important notes", "important_notes")]:
        if data[key]:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {x}" for x in data[key]))


def render_transport(data: dict):
    st.write(data["overview"])
    with st.expander(f"🚆 Intercity options ({data['origin']} → {data['destination']})"):
        for o in data["intercity_options"]:
            cost = f"{o['currency']} {o['estimated_cost']:,.0f}" if o["estimated_cost"] is not None else "price n/a"
            st.markdown(
                f"**{o['mode'].title()}** {o['origin']} → {o['destination']}"
                f"{' via ' + o['operator'] if o['operator'] else ''} · {o['duration']} · "
                f"{cost} ({o['price_status']})"
                + "".join(f"  \n[source]({u})" for u in o["source_urls"])
            )
    with st.expander("🚌 Local routes"):
        for r in data["local_routes"]:
            cost = f"{r['currency']} {r['estimated_cost']:,.0f}" if r["estimated_cost"] is not None else "price n/a"
            dist = f"{r['distance_km']:g} km · " if r["distance_km"] is not None else ""
            st.markdown(
                f"**{r['origin']} → {r['destination']}**: {r['recommended_transport']} · "
                f"{dist}{r['estimated_duration']} · {cost}"
                + "".join(f"  \n[source]({u})" for u in r["source_urls"])
            )
    for title, key in [("Entry points", "entry_points"),
                       ("Transport rules", "transport_rules"),
                       ("Important notes", "important_notes")]:
        if data[key]:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {x}" for x in data[key]))


def render_budget(data: dict):
    cur = data["currency"]
    st.write(data["overview"])
    cols = st.columns(3)
    cols[0].metric("Estimated total", f"{cur} {data['total_estimated_cost']:,.0f}" if data["total_estimated_cost"] is not None else "n/a")
    cols[1].metric("Per person", f"{cur} {data['cost_per_person']:,.0f}" if data["cost_per_person"] is not None else "n/a")
    if data["within_budget"] is not None:
        diff = data["budget_difference"]
        cols[2].metric("Budget", "Within budget" if data["within_budget"] else "Over budget",
                       f"{cur} {diff:,.0f}" if diff is not None else None)
    with st.expander("💰 Cost breakdown"):
        for c in data["cost_breakdown"]:
            cost = f"{c['currency']} {c['estimated_cost']:,.0f}" if c["estimated_cost"] is not None else "price n/a"
            st.markdown(
                f"**{c['category'].replace('_', ' ').title()}** – {c['description']} · {cost} ({c['price_status']})"
                + "".join(f"  \n[source]({u})" for u in c["source_urls"])
            )
    for title, key in [("Saving suggestions", "saving_suggestions"),
                       ("Important notes", "important_notes")]:
        if data[key]:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {x}" for x in data[key]))


def render_hotels(data: dict):
    cur = data["currency"]
    st.write(data["overview"])
    if data["estimated_accommodation_cost"] is not None:
        st.metric(f"Estimated accommodation ({data['number_of_nights']} nights)",
                  f"{cur} {data['estimated_accommodation_cost']:,.0f}")
    for region in data["regions"]:
        with st.expander(f"🏨 {region['region']}"):
            for h in region["hotels"]:
                price = f"{h['currency']} {h['price_per_night']:,.0f}/night" if h["price_per_night"] is not None else "price n/a"
                rating = f"⭐ {h['rating']}" + (f" ({h['rating_source']})" if h["rating_source"] else "") if h["rating"] is not None else "no rating"
                st.markdown(
                    f"**{h['name']}** – {h['property_type']}, {h['location']}  \n"
                    f"{h['description']}  \n"
                    f"{rating} · {price} ({h['price_status']})"
                    + (f"  \nAmenities: {', '.join(h['amenities'])}" if h["amenities"] else "")
                    + (f"  \nNearby: {', '.join(h['nearby_attractions'])}" if h["nearby_attractions"] else "")
                    + "".join(f"  \n[source]({u})" for u in h["source_urls"])
                )
    if data["important_notes"]:
        st.markdown("**Important notes**")
        st.markdown("\n".join(f"- {x}" for x in data["important_notes"]))


def render_itinerary(data: dict):
    cur = data["currency"]
    st.write(data["overview"])
    if data["total_estimated_cost"] is not None:
        st.metric("Total estimated cost", f"{cur} {data['total_estimated_cost']:,.0f}")
    for d in data["days"]:
        cost = f" · {cur} {d['estimated_cost']:,.0f}" if d["estimated_cost"] is not None else ""
        with st.expander(f"Day {d['day']}: {d['title']} – {d['region']}{cost}"):
            st.markdown("\n".join(f"- {a}" for a in d["activities"]))
            if d["accommodation"]:
                st.markdown(f"🏨 **Stay:** {d['accommodation']}")
            if d["transport"]:
                st.markdown(f"🚗 **Travel:** {d['transport']}")
    if data["important_notes"]:
        st.markdown("**Important notes**")
        st.markdown("\n".join(f"- {x}" for x in data["important_notes"]))


RENDERERS = {
    "destination": render_destination,
    "transport": render_transport,
    "hotel": render_hotels,
    "budget": render_budget,
    "itinerary": render_itinerary,
}


def render_result(agent_results: dict):
    for agent, res in agent_results.items():
        st.subheader(f"{agent.title()} results")
        RENDERERS[agent](res["data"])
        with st.expander(f"Raw {agent} data"):
            st.json(res["data"])


if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg.get("results"):
            render_result(msg["results"])
        else:
            st.write(msg["content"])

if prompt := st.chat_input("e.g. 7 days in Bhutan from Indore for 2 people, ₹1.5L, loves mountains"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # merge with what earlier messages already gave us, so follow-up answers aren't lost
        known = st.session_state.get("request", {})
        parsed = extractor.invoke(
            f"Details already collected: {known}\nUser's new message: {prompt}\n"
            "Return only what the user's new message states or corrects; leave other fields empty."
        ).model_dump(exclude_defaults=True, exclude_none=True)
        st.session_state.request = {**known, **parsed}
        req = ExtractedRequest(**st.session_state.request)
        missing = [f for f in ("origin", "destination", "duration_days", "travellers") if not getattr(req, f)]

        if missing:
            reply = f"I still need: {', '.join(missing)}. Could you share that?"
            st.write(reply)
            st.session_state.history.append({"role": "assistant", "content": reply})
        else:
            state = {
                "travel_request": req.model_dump(),
                "agent_tasks": {},
                "agent_results": {},
                "supervisor_decision": {},
                "workflow_status": "started",
                "clarification_question": "",
            }
            results = {}
            with st.status("Planning your trip...", expanded=True) as status:
                st.session_state.request = {}  # trip is planned; next message starts a new one
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # graph has a checkpointer
                for update in graph.stream(state, config, stream_mode="updates"):
                    for node, out in update.items():
                        st.write(f"✅ {AGENT_LABELS.get(node, node)}")
                        if node == "supervisor":
                            for agent, task in (out.get("agent_tasks") or {}).items():
                                st.caption(f"→ task for {agent}: {task[:200]}")
                        if out.get("agent_results"):
                            results = out["agent_results"]
                status.update(label="Planning complete", state="complete", expanded=False)

            if results:
                render_result(results)
                st.session_state.history.append({"role": "assistant", "results": results})
            else:
                reply = "No agent results were produced."
                st.write(reply)
                st.session_state.history.append({"role": "assistant", "content": reply})
