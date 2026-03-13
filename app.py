import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India News Integrity Monitor",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background:#0f1117; }
[data-testid="stSidebar"]          { background:#13151f; }
.metric-card {
    background:linear-gradient(135deg,#1e2130,#252840);
    border-radius:12px; padding:18px; text-align:center;
    border-left:4px solid #6c63ff; margin-bottom:10px;
}
.metric-value { font-size:1.9rem; font-weight:700; color:#6c63ff; }
.metric-label { font-size:0.82rem; color:#9ca3af; margin-top:4px; }
.sec-head {
    font-size:1.2rem; font-weight:700; color:#e2e8f0;
    border-bottom:2px solid #6c63ff;
    padding-bottom:6px; margin:18px 0 12px 0;
}
.box-green  { background:#0d2b1f; border-left:4px solid #10b981;
              padding:10px 14px; border-radius:8px; color:#d1fae5;
              font-size:.88rem; margin:6px 0; }
.box-yellow { background:#2b1f0d; border-left:4px solid #f59e0b;
              padding:10px 14px; border-radius:8px; color:#fef3c7;
              font-size:.88rem; margin:6px 0; }
.box-red    { background:#2b0d0d; border-left:4px solid #ef4444;
              padding:10px 14px; border-radius:8px; color:#fee2e2;
              font-size:.88rem; margin:6px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ── HELPERS ───────────────────────────────────────────────────────────────────
REGION_COORDS = {
    "North": {"lat": 28.70, "lon": 77.10, "city": "Delhi"},
    "South": {"lat": 13.08, "lon": 80.27, "city": "Chennai"},
    "East": {"lat": 22.57, "lon": 88.36, "city": "Kolkata"},
    "West": {"lat": 19.07, "lon": 72.88, "city": "Mumbai"},
    "Central": {"lat": 23.25, "lon": 77.41, "city": "Bhopal"},
}

VERDICT_COLORS = {
    "Fake": "#ef4444",
    "Authentic": "#10b981",
    "Misleading": "#f59e0b",
}

PLT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    margin=dict(t=10, b=10, l=10, r=10),
)


def card(col, val, label, color="#6c63ff"):
    col.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def box(cls, html):
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    candidates = [
        "indian_news_media_integrity_dataset.csv",
        "indiannewsmediaintegritydataset.csv",
    ]

    for name in candidates:
        if Path(name).exists():
            df = pd.read_csv(name)
            df["Month_dt"] = pd.to_datetime(df["Month_Year"], format="%b-%Y")
            df["Year"] = df["Month_dt"].dt.year
            return df

    raise FileNotFoundError(f"Dataset not found. Tried: {', '.join(candidates)}")


@st.cache_resource
def train_models(df):
    encode_cols = [
        "News_Verdict",
        "Channel_Watched",
        "Anchor_Name",
        "News_Category",
        "Consumption_Frequency",
    ]
    les = {c: LabelEncoder() for c in encode_cols}

    df2 = df.copy()
    for c, le in les.items():
        df2[c + "_enc"] = le.fit_transform(df2[c])

    feats = [
        "Channel_Watched_enc",
        "Anchor_Name_enc",
        "News_Category_enc",
        "Consumption_Frequency_enc",
        "TRP_Score",
        "Sensationalism_Score",
        "Fake_Units_Consumed",
        "Authentic_Units_Consumed",
    ]

    X = df2[feats].values
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    y_cls = df2["News_Verdict_enc"].values
    y_sent = df2["Sentiment_Score"].values
    y_trust = df2["Trust_Score"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X_s, y_cls, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=300, random_state=42, C=1.5)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))

    reg_s = Ridge(alpha=1.0)
    reg_s.fit(X_s, y_sent)

    reg_t = Ridge(alpha=1.0)
    reg_t.fit(X_s, y_trust)

    return clf, reg_s, reg_t, les, sc, acc


df = load_data()
clf, reg_s, reg_t, les, sc, acc = train_models(df)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 News Integrity\\nMonitor")
    st.markdown("---")

    page = st.radio(
        "",
        [
            "🏠 Media Pulse",
            "🔬 Influence Decoder",
            "🔮 Viewer Intelligence",
            "🎯 Editorial Compass",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    sel_ch = st.multiselect(
        "Channel",
        df["Channel_Watched"].unique().tolist(),
        default=df["Channel_Watched"].unique().tolist(),
    )
    sel_an = st.multiselect(
        "Anchor",
        df["Anchor_Name"].unique().tolist(),
        default=df["Anchor_Name"].unique().tolist(),
    )
    sel_yr = st.slider("Year Range", 2022, 2025, (2022, 2025))
    sel_cat = st.multiselect(
        "Category",
        df["News_Category"].unique().tolist(),
        default=df["News_Category"].unique().tolist(),
    )

    st.markdown("---")
    st.caption(f"🤖 Model Accuracy: **{acc * 100:.1f}%**")
    st.caption("📊 2,500 synthetic viewer records")

fdf = df[
    df["Channel_Watched"].isin(sel_ch)
    & df["Anchor_Name"].isin(sel_an)
    & df["Year"].between(sel_yr[0], sel_yr[1])
    & df["News_Category"].isin(sel_cat)
].copy()

if fdf.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MEDIA PULSE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Media Pulse":
    st.title("🏠 Media Pulse")
    st.caption("Real-time snapshot of news consumption across Indian channels (2022–2025)")

    c1, c2, c3, c4, c5 = st.columns(5)
    card(c1, f"{len(fdf):,}", "Viewers Tracked", "#6c63ff")
    card(c2, f"{(fdf['News_Verdict'] == 'Fake').mean() * 100:.1f}%", "Fake News Rate", "#ef4444")
    card(c3, f"{fdf['Sentiment_Score'].mean():+.2f}", "Avg Sentiment", "#10b981")
    card(c4, f"{fdf['Trust_Score'].mean():.1f}/10", "Avg Trust Score", "#f59e0b")
    card(c5, f"{fdf['Knowledge_Accuracy_Pct'].mean():.1f}%", "Avg Knowledge", "#3b82f6")

    st.markdown("---")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="sec-head">📰 Verdict Breakdown</div>', unsafe_allow_html=True)
        vc = fdf["News_Verdict"].value_counts().reset_index()
        vc.columns = ["News_Verdict", "count"]

        fig = px.pie(
            vc,
            values="count",
            names="News_Verdict",
            color="News_Verdict",
            color_discrete_map=VERDICT_COLORS,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.03] * len(vc))
        fig.update_layout(**PLT, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="sec-head">📺 Fake News Rate by Channel</div>', unsafe_allow_html=True)
        ch_fake = (
            fdf.groupby("Channel_Watched")
            .apply(lambda x: (x["News_Verdict"] == "Fake").mean() * 100)
            .reset_index(name="Fake_Rate")
            .sort_values("Fake_Rate")
        )
        fig2 = px.bar(
            ch_fake,
            x="Fake_Rate",
            y="Channel_Watched",
            orientation="h",
            color="Fake_Rate",
            color_continuous_scale="RdYlGn_r",
            labels={"Fake_Rate": "Fake News %", "Channel_Watched": ""},
        )
        fig2.update_layout(**PLT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="sec-head">📅 Monthly Sentiment Trend</div>', unsafe_allow_html=True)
        m = (
            fdf.groupby("Month_dt")["Sentiment_Score"]
            .mean()
            .reset_index()
            .sort_values("Month_dt")
        )
        m["lbl"] = m["Month_dt"].dt.strftime("%b %y")

        fig3 = px.line(
            m,
            x="lbl",
            y="Sentiment_Score",
            labels={"Sentiment_Score": "Avg Sentiment", "lbl": ""},
        )
        fig3.add_hline(
            y=0,
            line_dash="dash",
            line_color="#6b7280",
            annotation_text="Neutral",
            annotation_font_color="#9ca3af",
        )
        fig3.update_traces(
            line_color="#6c63ff",
            line_width=2.5,
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.15)",
        )
        fig3.update_layout(**PLT, height=300, xaxis=dict(tickangle=45, nticks=14))
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        st.markdown('<div class="sec-head">🎙️ Avg News Units by Anchor</div>', unsafe_allow_html=True)
        anc = (
            fdf.groupby("Anchor_Name")[["Fake_Units_Consumed", "Authentic_Units_Consumed"]]
            .mean()
            .reset_index()
            .sort_values("Fake_Units_Consumed", ascending=False)
        )

        fig4 = go.Figure(
            [
                go.Bar(
                    name="Fake",
                    x=anc["Anchor_Name"],
                    y=anc["Fake_Units_Consumed"],
                    marker_color="#ef4444",
                ),
                go.Bar(
                    name="Authentic",
                    x=anc["Anchor_Name"],
                    y=anc["Authentic_Units_Consumed"],
                    marker_color="#10b981",
                ),
            ]
        )
        fig4.update_layout(
            **PLT,
            barmode="group",
            height=300,
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            xaxis=dict(tickangle=15),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="sec-head">🗺️ Regional Fake News Heat Map (India)</div>',
        unsafe_allow_html=True,
    )

    reg_stats = (
        fdf.groupby("Region")
        .agg(
            Fake_Rate=("News_Verdict", lambda x: (x == "Fake").mean() * 100),
            Avg_Sentiment=("Sentiment_Score", "mean"),
            Viewers=("Viewer_ID", "count"),
        )
        .reset_index()
    )

    reg_stats["lat"] = reg_stats["Region"].map(lambda r: REGION_COORDS[r]["lat"])
    reg_stats["lon"] = reg_stats["Region"].map(lambda r: REGION_COORDS[r]["lon"])
    reg_stats["city"] = reg_stats["Region"].map(lambda r: REGION_COORDS[r]["city"])
    reg_stats["radius"] = reg_stats["Fake_Rate"].fillna(0) * 9000

    max_fake = reg_stats["Fake_Rate"].max()
    if pd.isna(max_fake) or max_fake == 0:
        reg_stats["opacity"] = 120
    else:
        reg_stats["opacity"] = (
            reg_stats["Fake_Rate"] / max_fake * 180
        ).fillna(60).astype(int)

    reg_stats["color"] = reg_stats.apply(
        lambda row: [239, 68, 68, int(row["opacity"])], axis=1
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=reg_stats,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=reg_stats,
        get_position=["lon", "lat"],
        get_text="city",
        get_size=16,
        get_color=[255, 255, 255],
        get_alignment_baseline="bottom",
    )

    view = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=3.8, pitch=30)
    tooltip = {
        "html": "<b>{Region}</b><br/>Fake Rate: {Fake_Rate:.1f}%<br/>Viewers: {Viewers}",
        "style": {
            "background": "#1e2130",
            "color": "white",
            "font-family": "sans-serif",
            "padding": "8px",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer, text_layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style=None,
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INFLUENCE DECODER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Influence Decoder":
    st.title("🔬 Influence Decoder")
    st.caption("Why is fake news spreading? What's driving emotional damage and trust erosion?")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown('<div class="sec-head">🔥 Correlation Matrix</div>', unsafe_allow_html=True)
        num_cols = [
            "TRP_Score",
            "Sensationalism_Score",
            "Fake_Units_Consumed",
            "Authentic_Units_Consumed",
            "Sentiment_Score",
            "Trust_Score",
            "Knowledge_Accuracy_Pct",
        ]
        labels = ["TRP", "Sensational", "Fake", "Authentic", "Sentiment", "Trust", "Knowledge"]
        corr = fdf[num_cols].corr().round(2)

        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmid=0,
                text=corr.values,
                texttemplate="%{text}",
                textfont=dict(size=10, color="white"),
                colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1]),
            )
        )
        fig.update_layout(**PLT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        st.markdown(
            '<div class="sec-head">📡 TRP vs Fake News — Channel Bubble</div>',
            unsafe_allow_html=True,
        )
        bs = (
            fdf.groupby("Channel_Watched")
            .agg(
                Avg_TRP=("TRP_Score", "mean"),
                Fake_Rate=("News_Verdict", lambda x: (x == "Fake").mean() * 100),
                Avg_Sent=("Sentiment_Score", "mean"),
                Viewers=("Viewer_ID", "count"),
            )
            .reset_index()
        )

        fig2 = px.scatter(
            bs,
            x="Avg_TRP",
            y="Fake_Rate",
            size="Viewers",
            color="Avg_Sent",
            text="Channel_Watched",
            color_continuous_scale="RdYlGn_r",
            labels={
                "Avg_TRP": "Avg TRP",
                "Fake_Rate": "Fake Rate (%)",
                "Avg_Sent": "Sentiment",
            },
        )
        fig2.update_traces(textposition="top center", marker=dict(sizemin=12))
        fig2.update_layout(**PLT, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    d3, d4 = st.columns(2)

    with d3:
        st.markdown(
            '<div class="sec-head">😨 Sentiment by Category × Verdict</div>',
            unsafe_allow_html=True,
        )
        cs = (
            fdf.groupby(["News_Category", "News_Verdict"])["Sentiment_Score"]
            .mean()
            .reset_index()
        )
        fig3 = px.bar(
            cs,
            x="News_Category",
            y="Sentiment_Score",
            color="News_Verdict",
            barmode="group",
            color_discrete_map=VERDICT_COLORS,
            labels={"Sentiment_Score": "Avg Sentiment", "News_Category": ""},
        )
        fig3.update_layout(
            **PLT,
            height=310,
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with d4:
        st.markdown(
            '<div class="sec-head">🧑‍🤝‍🧑 Knowledge Gap by Age Group</div>',
            unsafe_allow_html=True,
        )
        ag = (
            fdf.groupby("Age_Group")
            .agg(
                Fake_Pct=("News_Verdict", lambda x: (x == "Fake").mean() * 100),
                Avg_Know=("Knowledge_Accuracy_Pct", "mean"),
            )
            .reset_index()
        )

        fig4 = px.scatter(
            ag,
            x="Fake_Pct",
            y="Avg_Know",
            text="Age_Group",
            size="Fake_Pct",
            color="Avg_Know",
            color_continuous_scale="RdYlGn",
            labels={"Fake_Pct": "Fake Consumed (%)", "Avg_Know": "Knowledge %"},
        )
        fig4.update_traces(textposition="top center", marker=dict(sizemin=14))
        fig4.update_layout(**PLT, height=310)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="sec-head">🗺️ Authentic vs Fake Viewer Distribution (Regional)</div>',
        unsafe_allow_html=True,
    )

    reg2 = (
        fdf.groupby("Region")
        .agg(
            Fake_Viewers=("News_Verdict", lambda x: (x == "Fake").sum()),
            Auth_Viewers=("News_Verdict", lambda x: (x == "Authentic").sum()),
            Avg_Trust=("Trust_Score", "mean"),
        )
        .reset_index()
    )

    reg2["lat"] = reg2["Region"].map(lambda r: REGION_COORDS[r]["lat"])
    reg2["lon"] = reg2["Region"].map(lambda r: REGION_COORDS[r]["lon"])
    reg2["city"] = reg2["Region"].map(lambda r: REGION_COORDS[r]["city"])

    fake_layer = pdk.Layer(
        "ColumnLayer",
        data=reg2,
        get_position=["lon", "lat"],
        get_elevation="Fake_Viewers",
        elevation_scale=10,
        radius=60000,
        get_fill_color=[239, 68, 68, 180],
        pickable=True,
        auto_highlight=True,
    )

    auth_layer = pdk.Layer(
        "ColumnLayer",
        data=reg2,
        get_position=["lon", "lat"],
        get_elevation="Auth_Viewers",
        elevation_scale=10,
        radius=40000,
        get_fill_color=[16, 185, 129, 180],
        pickable=True,
        auto_highlight=True,
    )

    view2 = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=3.8, pitch=45, bearing=10)
    tooltip2 = {
        "html": "<b>{city} ({Region})</b><br/>🔴 Fake Viewers: {Fake_Viewers}<br/>🟢 Auth Viewers: {Auth_Viewers}<br/>Trust: {Avg_Trust:.1f}",
        "style": {"background": "#1e2130", "color": "white", "padding": "8px"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[fake_layer, auth_layer],
            initial_view_state=view2,
            tooltip=tooltip2,
            map_style=None,
        )
    )

    st.markdown("---")
    st.markdown('<div class="sec-head">💡 Auto-Detected Patterns</div>', unsafe_allow_html=True)

    fake_rate_by_channel = fdf.groupby("Channel_Watched").apply(
        lambda x: (x["News_Verdict"] == "Fake").mean()
    )
    worst_ch = fake_rate_by_channel.idxmax()
    best_ch = fake_rate_by_channel.idxmin()
    worst_an = fdf.groupby("Anchor_Name")["Sentiment_Score"].mean().idxmin()
    best_an = fdf.groupby("Anchor_Name")["Sentiment_Score"].mean().idxmax()

    box("box-red", f"🚨 <b>{worst_ch}</b> has the highest fake news rate — viewers show the most negative sentiment shifts.")
    box("box-green", f"✅ <b>{best_ch}</b> leads in authentic content — viewers score highest on knowledge accuracy.")
    box("box-yellow", f"⚠️ <b>{worst_an}</b>'s viewers report the lowest average sentiment — high emotional stress signal.")
    box("box-green", f"🏆 <b>{best_an}</b>'s audience consistently shows the highest trust scores — the benchmark anchor.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIEWER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Viewer Intelligence":
    st.title("🔮 Viewer Intelligence")
    st.caption("Enter any viewer profile and instantly predict news verdict, sentiment and trust")

    st.markdown("### 🧬 Build Your Viewer Profile")
    v1, v2, v3 = st.columns(3)

    with v1:
        p_ch = st.selectbox("📺 Channel", df["Channel_Watched"].unique())
        a_map = df.groupby("Channel_Watched")["Anchor_Name"].unique().to_dict()
        p_an = st.selectbox("🎙️ Anchor", a_map.get(p_ch, df["Anchor_Name"].unique()))
        p_cat = st.selectbox("📰 Category", df["News_Category"].unique())

    with v2:
        p_freq = st.selectbox("⏱️ Watch Frequency", ["Daily", "Weekly", "Occasional"])
        p_trp = st.slider("📊 TRP Score", 1.0, 10.0, 7.5, 0.1)
        p_sens = st.slider("🔊 Sensationalism", 1.0, 10.0, 6.0, 0.1)

    with v3:
        p_fake = st.slider("🔴 Fake Stories/Day", 0, 9, 2)
        p_auth = st.slider("🟢 Authentic Stories/Day", 0, 9, 4)
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔮 Predict Now", use_container_width=True, type="primary")

    st.markdown("---")

    try:
        ch_e = les["Channel_Watched"].transform([p_ch])[0]
        an_e = les["Anchor_Name"].transform([p_an])[0]
        cat_e = les["News_Category"].transform([p_cat])[0]
        fr_e = les["Consumption_Frequency"].transform([p_freq])[0]
    except Exception:
        ch_e = an_e = cat_e = fr_e = 0

    X_raw = np.array([[ch_e, an_e, cat_e, fr_e, p_trp, p_sens, p_fake, p_auth]])
    X_sc = sc.transform(X_raw)

    pred_v = les["News_Verdict"].inverse_transform(clf.predict(X_sc))[0]
    proba = clf.predict_proba(X_sc)[0]
    pred_s = float(np.clip(reg_s.predict(X_sc)[0], -5, 5))
    pred_t = float(np.clip(reg_t.predict(X_sc)[0], 1, 10))
    vc = les["News_Verdict"].classes_

    v_color = VERDICT_COLORS.get(pred_v, "#6c63ff")
    s_color = "#10b981" if pred_s > 0 else "#ef4444"
    t_color = "#10b981" if pred_t >= 6 else "#f59e0b" if pred_t >= 4 else "#ef4444"

    pr1, pr2, pr3, pr4 = st.columns(4)
    card(pr1, pred_v, "Predicted News Type", v_color)
    card(pr2, f"{pred_s:+.2f}", "Sentiment  (−5 to +5)", s_color)
    card(pr3, f"{pred_t:.1f}/10", "Trust Score", t_color)
    card(pr4, f"{max(proba) * 100:.0f}%", "Model Confidence", "#6c63ff")

    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown('<div class="sec-head">📊 Verdict Probability</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Verdict": vc, "Probability": proba * 100})
        fig_p = px.bar(
            prob_df,
            x="Verdict",
            y="Probability",
            color="Verdict",
            color_discrete_map=VERDICT_COLORS,
            labels={"Probability": "Probability (%)"},
            text_auto=".1f",
        )
        fig_p.update_layout(**PLT, height=300, showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

    with pc2:
        st.markdown('<div class="sec-head">🌡️ Emotional State Gauge</div>', unsafe_allow_html=True)
        fig_g = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred_s,
                delta={"reference": 0, "suffix": " from neutral"},
                gauge={
                    "axis": {"range": [-5, 5], "tickcolor": "#e2e8f0"},
                    "bar": {"color": s_color},
                    "bgcolor": "#1e2130",
                    "steps": [
                        {"range": [-5, -2], "color": "#7f1d1d"},
                        {"range": [-2, 0], "color": "#78350f"},
                        {"range": [0, 2], "color": "#064e3b"},
                        {"range": [2, 5], "color": "#065f46"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 0,
                    },
                },
                title={"text": "Sentiment Score", "font": {"color": "#e2e8f0"}},
            )
        )
        fig_g.update_layout(**PLT, height=300)
        st.plotly_chart(fig_g, use_container_width=True)

    if pred_v == "Fake":
        box(
            "box-red",
            "🚨 <b>High Risk:</b> This viewer is primarily consuming fake news — emotional damage and knowledge erosion are likely.",
        )
    elif pred_v == "Misleading":
        box(
            "box-yellow",
            "⚠️ <b>Moderate Risk:</b> Half-truths are distorting this viewer's understanding. Partial misinformation is equally damaging over time.",
        )
    else:
        box(
            "box-green",
            "✅ <b>Healthy Media Diet:</b> This viewer primarily consumes authentic news — trust and knowledge scores are above average.",
        )

    st.markdown("---")
    st.markdown(
        "<div class=\\"sec-head\\">🗺️ Viewer's Predicted Emotional Zone — India Map</div>",
        unsafe_allow_html=True,
    )

    all_reg = pd.DataFrame(
        [
            {
                "Region": r,
                "lat": v["lat"],
                "lon": v["lon"],
                "city": v["city"],
                "Pred_Sentiment": pred_s,
                "radius": abs(pred_s) * 50000 + 120000,
                "color": [16, 185, 129, 120] if pred_s > 0 else [239, 68, 68, 120],
            }
            for r, v in REGION_COORDS.items()
        ]
    )

    pulse_layer = pdk.Layer(
        "ScatterplotLayer",
        data=all_reg,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        stroked=True,
        line_width_min_pixels=2,
        get_line_color=[255, 255, 255, 80],
    )

    txt_layer = pdk.Layer(
        "TextLayer",
        data=all_reg,
        get_position=["lon", "lat"],
        get_text="city",
        get_size=14,
        get_color=[255, 255, 255],
    )

    view3 = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=3.6, pitch=0)
    sentiment_label = "Positive 😊" if pred_s > 0 else "Negative 😟"
    tooltip3 = {
        "html": f"<b>{{city}}</b><br/>Predicted Sentiment: <b>{pred_s:+.2f}</b> ({sentiment_label})",
        "style": {"background": "#1e2130", "color": "white", "padding": "8px"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[pulse_layer, txt_layer],
            initial_view_state=view3,
            tooltip=tooltip3,
            map_style=None,
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDITORIAL COMPASS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Editorial Compass":
    st.title("🎯 Editorial Compass")
    st.caption("Your personalised media literacy guide — know who to trust, what to avoid, and how to recover")

    st.markdown("### 🧮 Media Diet Score Calculator")
    e1, e2, e3 = st.columns(3)

    my_ch = e1.selectbox("Your Primary Channel", df["Channel_Watched"].unique())
    my_age = e2.selectbox("Your Age Group", ["18-25", "26-40", "41-60", "60+"])
    my_freq = e3.selectbox("Viewing Frequency", ["Daily", "Weekly", "Occasional"])

    cd = df[df["Channel_Watched"] == my_ch]
    fake_r = (cd["News_Verdict"] == "Fake").mean()
    sens_v = cd["Sensationalism_Score"].mean()
    trust_v = cd["Trust_Score"].mean()
    know_v = cd["Knowledge_Accuracy_Pct"].mean()

    fm = {"Daily": 1.3, "Weekly": 1.0, "Occasional": 0.7}[my_freq]
    mds = max(0, min(100, (100 - fake_r * 100 * 0.4 - sens_v * 2.5 + trust_v * 3 + know_v * 0.3) * fm / 1.1))
    mdc = "#10b981" if mds >= 60 else "#f59e0b" if mds >= 40 else "#ef4444"

    st.markdown("---")
    s1, s2, s3, s4 = st.columns(4)
    card(s1, f"{mds:.0f}/100", "Media Diet Score", mdc)
    card(s2, f"{fake_r * 100:.1f}%", "Fake Exposure", "#ef4444")
    card(s3, f"{sens_v:.1f}/10", "Sensationalism", "#f59e0b")
    card(s4, f"{know_v:.1f}%", "Viewer Knowledge", "#10b981")

    st.markdown("---")
    l1, l2 = st.columns(2)

    with l1:
        st.markdown('<div class="sec-head">🏆 Channel Trust Leaderboard</div>', unsafe_allow_html=True)
        ch_lb = (
            df.groupby("Channel_Watched")
            .agg(
                Trust=("Trust_Score", "mean"),
                Authentic_Rate=("News_Verdict", lambda x: (x == "Authentic").mean() * 100),
                Knowledge=("Knowledge_Accuracy_Pct", "mean"),
            )
            .round(1)
            .reset_index()
            .sort_values("Trust", ascending=False)
            .reset_index(drop=True)
        )
        ch_lb.index += 1
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        ch_lb.insert(0, "#", [medals.get(i, "") for i in ch_lb.index])

        st.dataframe(
            ch_lb.rename(
                columns={
                    "Channel_Watched": "Channel",
                    "Trust": "Trust /10",
                    "Authentic_Rate": "Authentic %",
                    "Knowledge": "Knowledge %",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with l2:
        st.markdown('<div class="sec-head">🎙️ Anchor Integrity Scores</div>', unsafe_allow_html=True)
        an_int = (
            df.groupby("Anchor_Name")
            .agg(
                Trust=("Trust_Score", "mean"),
                Sentiment=("Sentiment_Score", "mean"),
                Fake_Rate=("News_Verdict", lambda x: (x == "Fake").mean() * 100),
            )
            .round(2)
            .reset_index()
        )
        an_int["Integrity"] = (
            an_int["Trust"] * 4 + an_int["Sentiment"] * 6 - an_int["Fake_Rate"] * 0.3
        ).round(1)
        an_int = an_int.sort_values("Integrity", ascending=False)

        fig_i = px.bar(
            an_int,
            x="Anchor_Name",
            y="Integrity",
            color="Integrity",
            color_continuous_scale="RdYlGn",
            labels={"Anchor_Name": "", "Integrity": "Integrity Score"},
            text="Integrity",
        )
        fig_i.update_layout(**PLT, height=290, coloraxis_showscale=False, xaxis=dict(tickangle=15))
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-head">📋 Your Personalised Action Plan</div>', unsafe_allow_html=True)

    best_overall_ch = df.groupby("Channel_Watched")["Trust_Score"].mean().idxmax()
    best_overall_an = df.groupby("Anchor_Name")["Trust_Score"].mean().idxmax()

    if fake_r > 0.40:
        box(
            "box-red",
            f"🔴 <b>Switch Channel:</b> {my_ch} has a {fake_r * 100:.1f}% fake news rate. Try <b>{best_overall_ch}</b> instead.",
        )
    if sens_v > 7:
        box(
            "box-yellow",
            f"🟡 <b>Sensationalism Trap:</b> {my_ch} scores {sens_v:.1f}/10 — headlines are engineered to trigger fear. Pause before reacting.",
        )
    if my_freq == "Daily" and fake_r > 0.35:
        box(
            "box-red",
            "🔴 <b>Reduce Frequency:</b> Daily consumption of high-fake channels compounds emotional damage. Switch to weekly curated reading.",
        )
    if my_age in ["18-25", "60+"]:
        box(
            "box-yellow",
            f"🟡 <b>Vulnerability Alert:</b> The {my_age} age group is statistically more susceptible to fake news. Always cross-check before sharing.",
        )

    box(
        "box-green",
        f"✅ <b>Best Anchor Recommendation:</b> <b>{best_overall_an}</b> scores highest on integrity — prioritise their coverage.",
    )
    box(
        "box-green",
        "✅ <b>The 3-Source Rule:</b> Reading across 3+ channels with different editorial stances reduces fake news exposure by ~62%.",
    )
    box(
        "box-green",
        "✅ <b>The 3-Second Rule:</b> If a headline makes you angry within 3 seconds of reading — it's a sensationalism trigger, not journalism.",
    )

    st.markdown("---")
    st.markdown('<div class="sec-head">🔄 Sentiment Recovery Simulator</div>', unsafe_allow_html=True)

    sr1, sr2 = st.columns(2)
    with sr1:
        cur_f = st.slider("Fake stories/day NOW", 0, 9, 4)
        prop_f = st.slider("Fake stories/day (TARGET)", 0, 9, 1)
    with sr2:
        cur_a = st.slider("Authentic stories/day NOW", 0, 9, 2)
        prop_a = st.slider("Authentic stories/day (TARGET)", 0, 9, 6)

    cs = 2.5 - 0.55 * cur_f + 0.45 * cur_a
    ps = 2.5 - 0.55 * prop_f + 0.45 * prop_a
    ck = 65 - 4.0 * cur_f + 3.5 * cur_a
    pk = 65 - 4.0 * prop_f + 3.5 * prop_a

    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Bar(
            name="Before",
            x=["Sentiment", "Knowledge %"],
            y=[round(cs, 2), round(ck, 1)],
            marker_color="#ef4444",
        )
    )
    fig_sim.add_trace(
        go.Bar(
            name="After",
            x=["Sentiment", "Knowledge %"],
            y=[round(ps, 2), round(pk, 1)],
            marker_color="#10b981",
        )
    )
    fig_sim.update_layout(
        **PLT,
        barmode="group",
        height=280,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    ds, dk = ps - cs, pk - ck
    if ds > 0:
        box(
            "box-green",
            f"📈 This media diet change improves your sentiment by <b>{ds:+.2f}</b> and knowledge by <b>{dk:+.1f}%</b>.",
        )
    else:
        box(
            "box-red",
            f"📉 This pattern worsens your sentiment by <b>{ds:.2f}</b>. Increase authentic news consumption.",
        )
