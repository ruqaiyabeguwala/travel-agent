# Travel Agent

A multi-agent travel planner built with LangGraph and Streamlit. Describe a trip in plain English and a team of agents researches it, then a supervisor combines everything into a day-wise itinerary.

## How it works

```
supervisor → destination → transport → hotel → budget → itinerary
```

| Step | What it does |
|------|--------------|
| supervisor | Reads the request and plans the research |
| destination | Searches the web for regions, attractions, permits and seasonal notes |
| transport | Researches routes to and within the destination |
| hotel | Finds accommodation options with prices and ratings |
| budget | Searches for prices, totals the trip and checks it against your budget |
| itinerary | The supervisor receives the complete results of every agent and writes the final day-wise plan |

Web research uses Tavily; reasoning uses Google Gemini. Prices are marked as `estimate`, `live_verified` or `not_found`.

## Requirements

- Python 3.10 or newer (developed on 3.13)
- A [Google AI Studio](https://aistudio.google.com/) API key
- A [Tavily](https://tavily.com/) API key

## Install

```bash
cd "Travel Agent"

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configure

Create a `.env` file in this folder:

```
GOOGLE_API_KEY=your_google_ai_studio_key
TAVILY_API_KEY=your_tavily_key
```

Keep `.env` out of version control.

## Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), then describe your trip, for example:

> 7 days in Bhutan from Indore for 2 people, ₹1.5L budget, loves mountains and adventure

- If origin, destination, duration or number of travellers is missing, the bot asks for it. Your earlier answers are remembered.
- While planning, a status box shows each agent as it finishes.
- When it's done, every agent's results are shown as a summary with expandable details, ending with the day-wise itinerary.

A full run makes many web searches and model calls, so it can take a few minutes.

## Run without the UI

`main.py` runs the graph on a hardcoded sample trip (Indore → Bhutan) and prints progress to the terminal:

```bash
python main.py
```

Edit `initial_state` in `main.py` to try a different trip. The Streamlit app does not use it.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat UI |
| `main.py` | LangGraph definition and CLI entry point |
| `state.py` | Shared state, supervisor schema and all agent prompts |
| `supervisor.py` | Initial supervisor step and the final itinerary step |
| `destination_agent.py` | Destination research agent |
| `transport_agent.py` | Transport research agent |
| `hotel_agent.py` | Hotel research agent |
| `budget_agent.py` | Budget research agent |

## Adding an agent

1. Create `<name>_agent.py` with a response schema that includes an `overview` field, and a `<name>_node(state)` that returns `{"agent_results": {...}}`.
2. Add its prompt to `state.py`.
3. Register the node and edges in `main.py`.
4. In `app.py`, add a `render_<name>` function and register it in `RENDERERS`, plus a label in `AGENT_LABELS`.

## Troubleshooting

- **`Did not find tavily_api_key`**: `TAVILY_API_KEY` is missing from `.env`, or you started the app from a different folder.
- **Google API or model-not-found errors**: check `GOOGLE_API_KEY` and the model names in `supervisor.py` and the agent files.
- **`ModuleNotFoundError`**: the virtual environment isn't active, or `pip install -r requirements.txt` hasn't been run.
- **Streamlit shows old behaviour**: use "Rerun" or restart `streamlit run app.py`.
