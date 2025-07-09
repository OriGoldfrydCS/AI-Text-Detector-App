# ============================
# Import Setup
# ============================

import plotly.graph_objects as go

def gauge_chart(title: str, value: float, vmin=0, vmax=100, suffix=""):
    """
    Creates and returns a Plotly gauge chart.

    This function generates a styled gauge visualization with a numeric display
    and an optional suffix (e.g., %, ms). It's commonly used to show metric values
    within a fixed range, such as scores or performance indicators.

    Args:
        title (str): Title displayed above the gauge.
        value (float): The current value to show on the gauge.
        vmin (float): Minimum value on the gauge (default: 0).
        vmax (float): Maximum value on the gauge (default: 100).
        suffix (str): Optional string suffix to show after the value (e.g., '%').

    Returns:
        plotly.graph_objects.Figure: Configured gauge chart figure.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",                    # Display both the gauge and the number
            value=value,                            # Current value to display  
            number={'suffix': suffix},              # Optional suffix for unit (e.g., '%')
            gauge={
                'axis': {'range': [vmin, vmax]},    # Min and max bounds
                'bar': {'color': "#28666e"},      # Bar color for filled portion
                'bgcolor': "#162135"              # Gauge background color
            },
            title={'text': title, 'font': {'size': 14}}
        )
    )
    
    # Layout customization for dark theme
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),        # Padding around the figure
        paper_bgcolor="rgba(0,0,0,0)",              # Transparent background
        font_color="#ffffff",                     # White font color for visibility
        height=260                                  # Fixed height        
    )
    return fig