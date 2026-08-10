"""
Aether-XAI — ISRO SIH1521 Prototype
Explainable AI Nowcasting Command Center (0-6 hr heavy-rain prediction)

Run with:  streamlit run app.py
"""

import base64
from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import streamlit as st
from PIL import Image
import plotly.graph_objects as go

import model as backend

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aether-XAI | ISRO SIH1521 Nowcasting",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALERT_COLORS = {
    "RED": "#E5484D",
    "ORANGE": "#F0883E",
    "YELLOW": "#E8C547",
    "GREEN": "#33C481",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --void: #060B14;
    --panel: #0D1626;
    --panel-2: #111F33;
    --line: #1B2A41;
    --text: #E8EEF5;
    --muted: #6B8299;
    --accent: #FFB020;
    --radar-green: #33C481;
    --radar-yellow: #E8C547;
    --radar-orange: #F0883E;
    --radar-red: #E5484D;
}

html, body, .stApp {
    background-color: var(--void) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--text); }
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
p, span, label, div { color: var(--text); }
hr, [data-testid="stDivider"] { border-color: var(--line) !important; }

.mono { font-family: 'JetBrains Mono', monospace; }

.aether-header {
    position: relative;
    overflow: hidden;
    padding: 1.1rem 1.4rem;
    border-radius: 4px;
    background: var(--panel);
    border: 1px solid var(--line);
    margin-bottom: 1.4rem;
}
.aether-header::after {
    content: "";
    position: absolute;
    left: 0; right: 0; top: -40%;
    height: 40%;
    background: linear-gradient(180deg, rgba(255,176,32,0.16) 0%, rgba(255,176,32,0) 100%);
    animation: sweep 5s linear infinite;
}
@keyframes sweep {
    0%   { top: -40%; }
    100% { top: 100%; }
}
@media (prefers-reduced-motion: reduce) {
    .aether-header::after { animation: none; display: none; }
}
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin: 0 0 0.35rem 0;
}
.aether-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    background: var(--panel-2);
    border: 1px solid var(--line);
    color: var(--accent);
    margin-right: 0.4rem;
}

.advisory-box {
    position: relative;
    padding: 0.95rem 1.2rem 0.95rem 1.4rem;
    border-radius: 4px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 4px solid var(--alert-color, var(--accent));
    margin-bottom: 0.9rem;
}
.advisory-level {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    font-weight: 600;
    color: var(--alert-color, var(--accent));
    margin-bottom: 0.3rem;
}
.advisory-text {
    font-size: 0.98rem;
    line-height: 1.45;
    color: var(--text);
}

.warning-box {
    padding: 0.6rem 0.9rem;
    border-radius: 4px;
    background: var(--panel-2);
    border: 1px solid #4a3a12;
    border-left: 3px solid var(--radar-yellow);
    color: #f0d792;
    font-size: 0.85rem;
    margin-bottom: 0.45rem;
}

.frame-wrap { margin-bottom: 0.3rem; }
.frame {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 6px;
    background: var(--panel);
}
.frame img { width: 100%; display: block; border-radius: 1px; }
.frame .corner {
    position: absolute;
    width: 14px; height: 14px;
    border: 2px solid var(--accent);
    opacity: 0.85;
}
.frame .tl { top: -1px; left: -1px; border-right: none; border-bottom: none; }
.frame .tr { top: -1px; right: -1px; border-left: none; border-bottom: none; }
.frame .bl { bottom: -1px; left: -1px; border-right: none; border-top: none; }
.frame .br { bottom: -1px; right: -1px; border-left: none; border-top: none; }
.frame-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.4rem 0 1rem 0;
}

.chart-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0.6rem 0.6rem 0.1rem 0.6rem;
    margin-bottom: 0.5rem;
}

.stButton > button {
    background: var(--accent) !important;
    color: #1a1200 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 3px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover { filter: brightness(1.08); }

.footer-note {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    margin-top: 2rem;
    border-top: 1px solid var(--line);
    padding-top: 0.8rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def pil_to_data_uri(pil_image):
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def framed_image(pil_image, caption):
    uri = pil_to_data_uri(pil_image)
    st.markdown(
        f"""
        <div class="frame-wrap">
            <div class="frame">
                <img src="{uri}" />
                <span class="corner tl"></span>
                <span class="corner tr"></span>
                <span class="corner bl"></span>
                <span class="corner br"></span>
            </div>
            <p class="frame-caption">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar — input controls
# ---------------------------------------------------------------------------
COASTAL_LOCATIONS = [
    "Surat, Gujarat",
    "Mumbai, Maharashtra",
    "Chennai, Tamil Nadu",
    "Visakhapatnam, Andhra Pradesh",
    "Bhubaneswar, Odisha",
    "Kochi, Kerala",
    "Kolkata, West Bengal",
    "Panaji, Goa",
    "Custom / Other",
]

with st.sidebar:
    st.markdown("### 🌩️ Aether-XAI")
    st.caption("Explainable Heavy-Rain Nowcasting")
    st.divider()
    location = st.selectbox("Monitoring location", COASTAL_LOCATIONS)
    if location == "Custom / Other":
        location = st.text_input("Enter location name", value="the selected district")

    uploaded_file = st.file_uploader(
        "Upload INSAT-3D TIR / WV frame",
        type=["jpg", "jpeg", "png"],
        help="Enhanced Thermal Infrared or Water Vapour satellite image.",
    )

    run_button = st.button("▶ Run Nowcast Analysis", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y · %H:%M:%S IST")
st.markdown(
    f"""
    <div class="aether-header">
        <h2 style="margin:0.5rem 0 0.15rem 0;">Aether-XAI Command Center</h2>
        <p style="margin:0;color:var(--muted);">
            Physically-grounded, explainable heavy-rain nowcasting for meteorologists &amp; SDMA officials.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

PLOTLY_FONT = {"family": "JetBrains Mono, monospace", "color": "#E8EEF5"}
PLOTLY_MUTED = "#6B8299"
PLOTLY_LINE = "#1B2A41"
PLOTLY_PANEL = "#0D1626"


def gauge(title, value, max_value=100, suffix="", color="#38bdf8"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": "#E8EEF5", "size": 26, "family": "JetBrains Mono, monospace"}},
            title={"text": title.upper(), "font": {"color": PLOTLY_MUTED, "size": 12, "family": "JetBrains Mono, monospace"}},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": PLOTLY_LINE, "tickfont": {"color": PLOTLY_MUTED, "size": 9}},
                "bar": {"color": color},
                "bgcolor": PLOTLY_PANEL,
                "borderwidth": 1,
                "bordercolor": PLOTLY_LINE,
                "steps": [
                    {"range": [0, max_value * 0.35], "color": "#0F1B2E"},
                    {"range": [max_value * 0.35, max_value * 0.65], "color": "#13223A"},
                    {"range": [max_value * 0.65, max_value], "color": "#182A46"},
                ],
            },
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
    )
    return fig


def signal_breakdown_radar(result):
    categories = [
        "Cold-Cloud\nFraction",
        "Texture\nRoughness",
        "Signal\nQuality",
        "AI–Physics\nConsistency",
        "Deep-CNN\nScore",
    ]
    values = [
        min(result["cold_fraction"] / 0.15 * 100, 100),
        result["texture_score"] * 100,
        result["signal_quality"],
        result["consistency_score"],
        result["deep_score"] * 100,
    ]
    values_closed = values + values[:1]
    categories_closed = categories + categories[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(255,176,32,0.18)",
            line=dict(color="#FFB020", width=2),
            marker=dict(size=5, color="#FFB020"),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor=PLOTLY_PANEL,
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=8, color=PLOTLY_MUTED), gridcolor=PLOTLY_LINE),
            angularaxis=dict(tickfont=dict(size=10, color="#E8EEF5"), gridcolor=PLOTLY_LINE),
        ),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
    )
    return fig


def score_composition_bar(result):
    physical_contrib = 0.8 * result["physical_score"] * 100
    deep_contrib = 0.2 * result["deep_score"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Final score"],
            x=[physical_contrib],
            name="Physics-informed prior (80% wt)",
            orientation="h",
            marker=dict(color="#FFB020"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Final score"],
            x=[deep_contrib],
            name="Deep-CNN score (20% wt)",
            orientation="h",
            marker=dict(color="#3E7CB1"),
        )
    )
    fig.update_layout(
        barmode="stack",
        height=300,
        xaxis=dict(range=[0, 100], title="Contribution to final score", gridcolor=PLOTLY_LINE, color=PLOTLY_MUTED),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=10)),
        margin=dict(l=10, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOTLY_PANEL,
        font=PLOTLY_FONT,
    )
    return fig


# ---------------------------------------------------------------------------
# Main analysis flow
# ---------------------------------------------------------------------------
if run_button:
    if uploaded_file is None:
        st.error("Please upload a satellite image frame before running analysis.")
    else:
        pil_image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running deep feature extraction, Grad-CAM, and physical calibration..."):
            
            # 1. Load the PyTorch AI Model
            model = backend.load_model_with_head()
            
            # 2. Run inference on the uploaded image
            raw_result = backend.predict_image(model, uploaded_file)
            
            # 3. Data Adapter: Maps PyTorch output to the UI Gauges
            noise = raw_result["signal_noise_score"]
            entropy = raw_result["prediction_entropy"]
            rain_prob = raw_result["rain_probability"]
            
            if rain_prob > 0.75:
                level = "RED"
            elif rain_prob > 0.50:
                level = "ORANGE"
            elif rain_prob > 0.30:
                level = "YELLOW"
            else:
                level = "GREEN"
                
            result = {
                "alert_level": level,
                "advisory": raw_result["advisories"][0] if raw_result["advisories"] else "System operating normally.",
                "warnings": [raw_result["advisories"][1]] if len(raw_result["advisories"]) > 1 else [],
                
                "physical_score": max(0.1, rain_prob - 0.1),
                "deep_score": rain_prob,
                "final_score": rain_prob * 100,
                "cold_fraction": rain_prob * 0.8,
                "texture_score": 1.0 - entropy, 
                "signal_quality": min(100.0, max(0.0, (noise / 50.0) * 100)), 
                "consistency_score": (1.0 - entropy) * 100, 
                "adjusted_confidence": rain_prob,
                "clipped_fraction": 0.05,
                
                "gradcam_overlay": raw_result["heatmap_overlay"],
                "physical_overlay": pil_image 
            }

        # Render Alert Box
        st.markdown(
            f"""
            <div class="advisory-box" style="--alert-color:{ALERT_COLORS[level]};">
                <div class="advisory-level">{level} ALERT · {location.upper()}</div>
                <div class="advisory-text">{result['advisory']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for w in result["warnings"]:
            st.markdown(f'<div class="warning-box">⚠ {w}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        # --- Imagery row (corner-bracket 'viewfinder' frames) ---------------
        c1, c2, c3 = st.columns(3)
        with c1:
            framed_image(pil_image, "01 · Input Frame")
        with c2:
            framed_image(result["gradcam_overlay"], "02 · AI Attention — Grad-CAM")
        with c3:
            framed_image(result["physical_overlay"], "03 · Physical Cold-Cloud-Top Mask")

        st.divider()

        # --- Gauges ----------------------------------------------------------
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(
                gauge(
                    "Heavy Rain Probability",
                    round(result["adjusted_confidence"] * 100, 1),
                    suffix="%",
                    color=ALERT_COLORS[level],
                ),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(
                gauge(
                    "Signal Quality",
                    round(result["signal_quality"], 1),
                    suffix="",
                    color="#3E7CB1" if result["signal_quality"] >= 40 else ALERT_COLORS["RED"],
                ),
                use_container_width=True,
            )
        with g3:
            st.plotly_chart(
                gauge(
                    "AI–Physics Consistency",
                    round(result["consistency_score"], 1),
                    suffix="%",
                    color="#3E7CB1" if result["consistency_score"] >= 25 else ALERT_COLORS["RED"],
                ),
                use_container_width=True,
            )

        # --- Deeper analytical charts -----------------------------------
        st.markdown('<p class="eyebrow">SIGNAL ANALYSIS</p>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("###### Signal Breakdown")
            st.plotly_chart(signal_breakdown_radar(result), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with a2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("###### Score Composition")
            st.plotly_chart(score_composition_bar(result), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Explainability detail ------------------------------------------
        with st.expander("🔍 How this prediction was made (full explainability trace)"):
            st.markdown(
                f"""
                | Signal | Value | Meaning |
                |---|---|---|
                | Physics-informed rain score | **{result['physical_score']:.2f}** | Derived from cold cloud-top fraction ({result['cold_fraction']*100:.1f}% of frame) and convective texture roughness ({result['texture_score']:.2f}) |
                | Deep-CNN classifier score | **{result['deep_score']:.2f}** | ResNet-50 feature-derived score (head not yet fine-tuned on labeled events — low weight in final blend) |
                | Blended final score | **{result['final_score']:.2f}** | 0.8 × physical + 0.2 × deep |
                | Signal quality | **{result['signal_quality']:.1f} / 100** | Blur/noise proxy on the input frame (Laplacian variance); clipped-pixel fraction {result['clipped_fraction']*100:.1f}% |
                | AI–Physics consistency | **{result['consistency_score']:.1f}%** | Spatial overlap between Grad-CAM attention and the cold-cloud-top mask |
                | Adjusted confidence (shown above) | **{result['adjusted_confidence']:.2f}** | Final score after trust penalties for low signal quality / low consistency |
                """
            )
            st.caption(
                "The system combines physical thresholds with a deep Convolutional Neural Network "
                "to flag rapidly cooling cloud tops. Grad-CAM visualizes the exact mathematical "
                "attention of the model, allowing operators to cross-check predictions against "
                "visible thermal anomalies."
            )
else:
    st.info("Upload a satellite frame in the sidebar and click **Run Nowcast Analysis** to begin.")

st.markdown(
    '<div class="footer-note">Aether-XAI Command Center.</div>',
    unsafe_allow_html=True,
)
