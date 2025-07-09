# =======================
# Import Setup
# =======================

import os
import sys

# Add project root to Python path for relative imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Dash and Plotly for UI and visualizations
import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from detection import analyze_text

# =======================
# App Initialization
# =======================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# =======================
# Styling Constants
# =======================

BASE_CARD_STYLE = {
    "backgroundColor": "#1b2638",
    "borderRadius": "8px",
    "border": "1px solid #2c3e50"
}

# =======================
# Gauge Utility Functions
# =======================

def empty_gauge(title, vmin, vmax):
    """
    Creates an empty gauge chart with no value filled.

    Parameters:
        title (str): Title of the gauge.
        vmin (float): Minimum value of the gauge range.
        vmax (float): Maximum value of the gauge range.

    Returns:
        go.Figure: A Plotly gauge chart object.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=vmin,
        gauge={
            "axis": {"range": [vmin, vmax], "tickcolor": "#ffffff"},
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "#162135",
            "borderwidth": 0
        },
        title={"text": title, "font": {"size": 14}}
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        height=260
    )
    return fig

def make_gauge(title, value, vmin, vmax, color, suffix=""):
    """
    Creates a filled gauge chart with a specified value.

    Parameters:
        title (str): Gauge title.
        value (float): Current value.
        vmin (float): Min axis value.
        vmax (float): Max axis value.
        color (str): Color of the bar.
        suffix (str): Suffix for the number display (e.g., '%').

    Returns:
        go.Figure: A Plotly gauge chart with a filled value.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"color": color},
            "bgcolor": "#162135",
            "borderwidth": 0
        },
        title={"text": title, "font": {"size": 14}}
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        height=260
    )
    return fig

# =======================
# App Layout
# =======================

app.layout = dbc.Container(fluid=True, style={"padding": "2rem"}, children=[
    
    # Header
    html.Div(id="header", children=[
        html.H1("AI Text Detection"),
        html.P("Advanced machine learning analysis to detect AI-generated content with precision")
    ]),

    # Input and results row
    dbc.Row([
        # Text input section
        dbc.Col(html.Div(id="text-input-box", children=[
            html.H4("📝 Text Analysis"),
            dcc.Textarea(
                id="input-text",
                placeholder="Paste or type text to analyze",
                style={"width": "100%", "height": "150px"},
                value=""
            ),
            html.Div(id="char-word-count", style={"margin": "0.5rem 0"}),
            dbc.Button("Analyze Text", id="analyze-button", color="primary"),
            html.Hr(),
        ]), width=6),

        # Results display section
        dbc.Col(html.Div(id="results-box", children=[
            html.H4("Detection Results"),
            dbc.Row([
                # Human result card
                dbc.Col(dbc.Card(id="human-card", style=BASE_CARD_STYLE, children=[
                    dbc.CardBody([
                        html.H5("👤 Human Written", className="card-title"),
                        html.P("Natural language patterns"),
                        html.H2(id="human-score", style={"color": "#00cc96"}),
                        html.Div(id="human-bar")
                    ])
                ]), width=6),

                # AI result card
                dbc.Col(dbc.Card(id="ai-card", style=BASE_CARD_STYLE, children=[
                    dbc.CardBody([
                        html.H5("💻 AI Generated", className="card-title"),
                        html.P("Machine learning patterns"),
                        html.H2(id="ai-score", style={"color": "#EF553B"}),
                        html.Div(id="ai-bar")
                    ])
                ]), width=6),
            ]),
            html.Div(id="confidence-level",
                     style={"marginTop": "1rem", "fontStyle": "italic", "color": "#e74c3c"})
        ]), width=6),
    ]),

    html.Hr(),

    # Metrics Dashboard
    html.Div(id="metrics-box", children=[
        html.H4("Analysis Metrics Dashboard"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="gauge-perplexity",
                              figure=empty_gauge("Perplexity", 0, 200)), width=2),
            dbc.Col(dcc.Graph(id="gauge-ttr",
                              figure=empty_gauge("Type-Token Ratio", 0, 100)), width=2),
            dbc.Col(dcc.Graph(id="gauge-rep",
                              figure=empty_gauge("Repetition Rate", 0, 100)), width=2),
            dbc.Col(dcc.Graph(id="gauge-avg-sent",
                              figure=empty_gauge("Avg Sentence Length", 0, 50)), width=2),
            dbc.Col(dcc.Graph(id="gauge-avg-word",
                              figure=empty_gauge("Avg Word Length", 0, 50)), width=2),
        ], justify="around")
    ])
])

# =======================
# Callbacks
# =======================

@app.callback(
    Output("char-word-count", "children"),
    Input("input-text", "value")
)

def update_counts(text):
    """
    Displays the current character and word count of the text input.
    """
    text = text or ""
    return f"{len(text)} characters · {len(text.split())} words"

@app.callback(
    Output("input-text", "value"),
    Input({"type": "sample-texts", "index": ALL}, "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True
)
def fill_sample_text(n_clicks, current):
    """
    Fills the input box with one of several predefined sample texts.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    prop_id = ctx.triggered[0]["prop_id"]
    idx = int(prop_id.split('index":')[1].split('}')[0])
    samples = [
        "The quick brown fox jumps over the lazy dog. This is a simple simple sentence that demonstrates basic English vocabulary and structure.",
        "In accordance with the provisions outlined in the aforementioned documentation, the committee has resolved to implement the proposed amendments.",
        "I absolutely love spending time in nature! There's something wonderful wonderful about hiking through lush forests and breathing in the fresh air."
    ]
    return samples[idx]

@app.callback(
    Output("human-score", "children"),
    Output("ai-score", "children"),
    Output("confidence-level", "children"),
    Output("gauge-perplexity", "figure"),
    Output("gauge-ttr", "figure"),
    Output("gauge-rep", "figure"),
    Output("gauge-avg-sent", "figure"),
    Output("gauge-avg-word", "figure"),  # Ensure this matches the metric name
    Output("human-bar", "children"),
    Output("ai-bar", "children"),
    Output("human-card", "style"),
    Output("ai-card", "style"),
    Input("analyze-button", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True
)
def analyze(n_clicks, text):
    """
    Main callback for analyzing input text using ML-based detection.

    Returns updated UI elements showing:
    - human/AI probabilities
    - per-metric gauges
    - progress bars
    - result message and highlight styles
    """
    text = (text or "").strip()
    if not text:
        msg = "Please paste or type some text above."
        return (
            "—", "—", msg,
            empty_gauge("Perplexity", 0, 200),
            empty_gauge("Type-Token Ratio", 0, 100),
            empty_gauge("Repetition Rate", 0, 100),
            empty_gauge("Avg Sentence Length", 0, 50),
            empty_gauge("Avg Word Length", 0, 50),  # Ensure this matches
            "", "",
            BASE_CARD_STYLE, BASE_CARD_STYLE
        )

    try:
        human_prob, ai_prob, metrics = analyze_text(text)
    except Exception as e:
        print(f"Error in analyze_text(): {e}", file=sys.stderr)
        err = f"Analysis error: {e}"
        return (
            "Error", "Error", err,
            empty_gauge("Perplexity", 0, 200),
            empty_gauge("Type-Token Ratio", 0, 100),
            empty_gauge("Repetition Rate", 0, 100),
            empty_gauge("Avg Sentence Length", 0, 50),
            empty_gauge("Avg Word Length", 0, 50),  # Ensure this matches
            "", "",
            BASE_CARD_STYLE, BASE_CARD_STYLE
        )

    # Format percentage strings
    hp, ap = f"{human_prob*100:.1f}%", f"{ai_prob*100:.1f}%"

    # Decision threshold
    THRESHOLD = 0.8  
    if human_prob >= THRESHOLD and human_prob > ai_prob:
        result_text = f"Detected: Human ({hp} vs {ap})"
        human_style = {**BASE_CARD_STYLE, "border": "2px solid #00cc96"}
        ai_style = BASE_CARD_STYLE
    elif ai_prob >= THRESHOLD and ai_prob > human_prob:
        result_text = f"Detected: AI ({hp} vs {ap})"
        human_style = BASE_CARD_STYLE
        ai_style = {**BASE_CARD_STYLE, "border": "2px solid #EF553B"}
    else:
        result_text = f"Detected: Tie/Uncertain ({hp} vs {ap})"
        human_style = ai_style = BASE_CARD_STYLE

    # Build gauges for metrics
    g1 = make_gauge("Perplexity", metrics["Perplexity"], 0, 200, "#EF553B", "")
    g2 = make_gauge("Type-Token Ratio", metrics["Type-Token Ratio"], 0, 100, "#00CC96", "%")
    g3 = make_gauge("Repetition Rate", metrics["Repetition Rate"], 0, 100, "#FFA15A", "%")
    g4 = make_gauge("Avg Sentence Length", metrics["Avg Sentence Length"], 0, 50, "#AB63FA", "")
    g5 = make_gauge("Avg Word Length", metrics["Avg Word Length"], 0, 50, "#19D3F3", "")  

    # Progress bars
    human_bar = dbc.Progress(value=human_prob*100, color="success", style={"height": "8px"}, className="mt-2")
    ai_bar = dbc.Progress(value=ai_prob*100, color="danger", style={"height": "8px"}, className="mt-2")

    return (
        hp, ap, result_text,
        g1, g2, g3, g4, g5,
        human_bar, ai_bar,
        human_style, ai_style
    )

# =======================
# Run App
# =======================

if __name__ == "__main__":
    app.run(debug=True)