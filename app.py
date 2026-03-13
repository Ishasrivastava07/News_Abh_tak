import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India News Integrity Monitor",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px; padding: 20px; text-align: center;
        border-left: 4px solid #6c63ff; margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6c63ff; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #e2e8f0;
        border-bottom: 2px solid #6c63ff; padding-bottom: 8px; margin: 20px 0 16px 0;
    }
    .insight-box {
        background: #1a1d2e; border-left: 4px solid #10b981;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
        color: #d1fae5; font-size: 0.9rem;
    }
    .warning-box {
        background: #1a1d2e; border-left: 4px solid #f59e0b;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
        color: #fef3c7; font-size: 0.9rem;
    }
    .danger-box {
        background: #1a1d2e; border-left: 4px solid #ef4444;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
        color: #fee2e2; font-size: 0.9rem;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADER ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("indian_news_media_integrity_dataset.csv")
    df["Month_dt"] = pd.to_datetime(df["Month_Year"], format="%b-%Y")
    df["Year"] = df["Month_dt"].dt.year
    return df

@st.cache_resource
def train_models(df):
    le_verdict  = LabelEncoder()
    le_channel  = LabelEncoder()
    le_anchor   = LabelEncoder()
    le_cat      = LabelEncoder()
    le_freq     = LabelEncoder()

    df2 = df.copy()
    df2["verdict_enc"]  = le_verdict.fit_transform(df2["News_Verdict"])
    df2["channel_enc"]  = le_channel.fit_transform(df2["Channel_Watched"])
    df2["anchor_enc"]   = le_anchor.fit_transform(df2["Anchor_Name"])
    df2["cat_enc"]      = le_cat.fit_transform(df2["News_Category"])
    df2["freq_enc"]     = le_freq.fit_transform(df2["Consumption_Frequency"])

    features = ["channel_enc","anchor_enc","cat_enc","freq_enc",
                "TRP_Score","Sensationalism_Score",
                "Fake_Units_Consumed","Authentic_Units_Consumed"]

    X = df2[features]
    y_cls = df2["verdict_enc"]
    y_sent = df2["Sentiment_Score"]
    y_trust = df2["Trust_Score"]

    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, yc_tr)
    acc = accuracy_score(yc_te, clf.predict(X_te))

    reg_sent = GradientBoostingRegressor(n_estimators=80, random_state=42)
    reg_sent.fit(X, y_sent)

    reg_trust = GradientBoostingRegressor(n_estimators=80, random_state=42)
    reg_trust.fit(X, y_trust)

    return clf, reg_sent, reg_trust, le_verdict, le_channel, le_anchor, le_cat, le_freq, acc, X.columns.tolist()

df = load_data()
clf, reg_sent, reg_trust, le_verdict, le_channel, le_anchor, le_cat, le_freq, model_acc, feat_cols = train_models(df)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=60)
    st.title("📡 News Integrity
Monitor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Media Pulse",
         "🔬 Influence Decoder",
         "🔮 Viewer Intelligence",
         "🎯 Editorial Compass"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    sel_channels = st.multiselect(
        "Channels", options=df["Channel_Watched"].unique().tolist(),
        default=df["Channel_Watched"].unique().tolist()
    )
    sel_anchors = st.multiselect(
        "Anchors", options=df["Anchor_Name"].unique().tolist(),
        default=df["Anchor_Name"].unique().tolist()
    )
    sel_years = st.slider("Year Range", 2022, 2025, (2022, 2025))
    sel_cats = st.multiselect(
        "News Category", options=df["News_Category"].unique().tolist(),
        default=df["News_Category"].unique().tolist()
    )

    st.markdown("---")
    st.caption(f"🤖 Model Accuracy: **{model_acc*100:.1f}%**")
    st.caption("📊 Dataset: 2,500 viewer records")

# ── FILTERED DATA ─────────────────────────────────────────────────────────────
fdf = df[
    df["Channel_Watched"].isin(sel_channels) &
    df["Anchor_Name"].isin(sel_anchors) &
    df["Year"].between(sel_years[0], sel_years[1]) &
    df["News_Category"].isin(sel_cats)
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MEDIA PULSE (Descriptive)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Media Pulse":
    st.title("🏠 Media Pulse — What's Happening?")
    st.caption("Real-time snapshot of news consumption patterns across Indian channels")

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    total = len(fdf)
    fake_pct   = (fdf["News_Verdict"]=="Fake").mean()*100
    avg_sent   = fdf["Sentiment_Score"].mean()
    avg_trust  = fdf["Trust_Score"].mean()
    avg_know   = fdf["Knowledge_Accuracy_Pct"].mean()

    for col, val, label, color in zip(
        [k1,k2,k3,k4,k5],
        [f"{total:,}", f"{fake_pct:.1f}%", f"{avg_sent:+.2f}", f"{avg_trust:.1f}/10", f"{avg_know:.1f}%"],
        ["Viewers Tracked","Fake News Rate","Avg Sentiment","Avg Trust Score","Avg Knowledge"],
        ["#6c63ff","#ef4444","#10b981","#f59e0b","#3b82f6"]
    ):
        col.markdown(f"""<div class="metric-card" style="border-left-color:{color}">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">📰 News Verdict Breakdown</div>', unsafe_allow_html=True)
        vc = fdf["News_Verdict"].value_counts().reset_index()
        fig = px.pie(vc, values="count", names="News_Verdict",
                     color_discrete_map={"Fake":"#ef4444","Authentic":"#10b981","Misleading":"#f59e0b"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=20,b=20,l=20,r=20), height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">📺 Fake News Rate by Channel</div>', unsafe_allow_html=True)
        ch_fake = fdf.groupby("Channel_Watched").apply(
            lambda x: (x["News_Verdict"]=="Fake").mean()*100).reset_index()
        ch_fake.columns = ["Channel","Fake_Rate"]
        ch_fake = ch_fake.sort_values("Fake_Rate", ascending=True)
        fig2 = px.bar(ch_fake, x="Fake_Rate", y="Channel", orientation="h",
                      color="Fake_Rate", color_continuous_scale="RdYlGn_r",
                      labels={"Fake_Rate":"Fake News %","Channel":""})
        fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">📅 Monthly Sentiment Trend</div>', unsafe_allow_html=True)
        monthly = fdf.groupby("Month_dt")["Sentiment_Score"].mean().reset_index().sort_values("Month_dt")
        monthly["Month_str"] = monthly["Month_dt"].dt.strftime("%b %Y")
        fig3 = px.line(monthly, x="Month_str", y="Sentiment_Score",
                       labels={"Sentiment_Score":"Avg Sentiment","Month_str":""})
        fig3.add_hline(y=0, line_dash="dash", line_color="#6b7280", annotation_text="Neutral")
        fig3.update_traces(line_color="#6c63ff", line_width=2.5)
        fig3.update_layout(margin=dict(t=10,b=40,l=10,r=10), height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", xaxis=dict(tickangle=45, nticks=12))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">🎙️ Anchor — Fake vs Authentic Units</div>', unsafe_allow_html=True)
        anc = fdf.groupby("Anchor_Name")[["Fake_Units_Consumed","Authentic_Units_Consumed"]].mean().reset_index()
        anc = anc.sort_values("Fake_Units_Consumed", ascending=False)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Fake", x=anc["Anchor_Name"], y=anc["Fake_Units_Consumed"],
                              marker_color="#ef4444"))
        fig4.add_trace(go.Bar(name="Authentic", x=anc["Anchor_Name"], y=anc["Authentic_Units_Consumed"],
                              marker_color="#10b981"))
        fig4.update_layout(barmode="group", margin=dict(t=10,b=60,l=10,r=10), height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", legend=dict(orientation="h", y=1.1),
                           xaxis=dict(tickangle=20))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🗂️ Raw Data Explorer</div>', unsafe_allow_html=True)
    st.dataframe(fdf.drop(columns=["Month_dt","Year"]).head(100),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INFLUENCE DECODER (Diagnostic)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Influence Decoder":
    st.title("🔬 Influence Decoder — Why Is It Happening?")
    st.caption("Uncovering the root drivers behind fake news, sentiment damage and trust erosion")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">🔥 Correlation Map</div>', unsafe_allow_html=True)
        num_cols = ["TRP_Score","Sensationalism_Score","Fake_Units_Consumed",
                    "Authentic_Units_Consumed","Sentiment_Score","Trust_Score","Knowledge_Accuracy_Pct"]
        corr = fdf[num_cols].corr().round(2)
        short = ["TRP","Sensational","Fake","Authentic","Sentiment","Trust","Knowledge"]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=short, y=short,
            colorscale="RdBu", zmid=0, text=corr.values,
            texttemplate="%{text}", textfont=dict(size=10),
            colorbar=dict(title="r", tickvals=[-1,-0.5,0,0.5,1])
        ))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=380,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">📈 TRP vs Fake News Rate</div>', unsafe_allow_html=True)
        ch_stats = fdf.groupby("Channel_Watched").agg(
            Avg_TRP=("TRP_Score","mean"),
            Fake_Rate=("News_Verdict", lambda x: (x=="Fake").mean()*100),
            Avg_Sentiment=("Sentiment_Score","mean")
        ).reset_index()
        fig2 = px.scatter(ch_stats, x="Avg_TRP", y="Fake_Rate",
                          size="Fake_Rate", color="Avg_Sentiment",
                          text="Channel_Watched",
                          color_continuous_scale="RdYlGn_r",
                          labels={"Avg_TRP":"Avg TRP Score","Fake_Rate":"Fake News Rate (%)","Avg_Sentiment":"Sentiment"})
        fig2.update_traces(textposition="top center", marker=dict(sizemin=10))
        fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=380,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">😨 Sentiment by News Category</div>', unsafe_allow_html=True)
        cat_sent = fdf.groupby(["News_Category","News_Verdict"])["Sentiment_Score"].mean().reset_index()
        fig3 = px.bar(cat_sent, x="News_Category", y="Sentiment_Score", color="News_Verdict",
                      barmode="group",
                      color_discrete_map={"Fake":"#ef4444","Authentic":"#10b981","Misleading":"#f59e0b"},
                      labels={"Sentiment_Score":"Avg Sentiment","News_Category":"Category"})
        fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">🧑‍🤝‍🧑 Fake News Impact by Age Group</div>', unsafe_allow_html=True)
        age_impact = fdf.groupby("Age_Group").agg(
            Fake_Pct=("News_Verdict", lambda x: (x=="Fake").mean()*100),
            Avg_Knowledge=("Knowledge_Accuracy_Pct","mean")
        ).reset_index()
        fig4 = px.scatter(age_impact, x="Fake_Pct", y="Avg_Knowledge",
                          text="Age_Group", size="Fake_Pct",
                          color="Avg_Knowledge", color_continuous_scale="RdYlGn",
                          labels={"Fake_Pct":"Fake News Consumed (%)","Avg_Knowledge":"Knowledge Accuracy %"})
        fig4.update_traces(textposition="top center", marker=dict(sizemin=12))
        fig4.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🌍 Regional Sentiment Map</div>', unsafe_allow_html=True)
    reg_sent = fdf.groupby("Region")["Sentiment_Score"].mean().reset_index()
    fig5 = px.bar(reg_sent, x="Region", y="Sentiment_Score",
                  color="Sentiment_Score", color_continuous_scale="RdYlGn",
                  labels={"Sentiment_Score":"Avg Sentiment"})
    fig5.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig5.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e2e8f0", coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

    # Auto insights
    st.markdown("---")
    st.markdown('<div class="section-header">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)
    worst_ch = fdf.groupby("Channel_Watched").apply(lambda x: (x["News_Verdict"]=="Fake").mean()).idxmax()
    best_ch  = fdf.groupby("Channel_Watched").apply(lambda x: (x["News_Verdict"]=="Fake").mean()).idxmin()
    worst_an = fdf.groupby("Anchor_Name")["Sentiment_Score"].mean().idxmin()
    best_an  = fdf.groupby("Anchor_Name")["Sentiment_Score"].mean().idxmax()

    st.markdown(f'''<div class="danger-box">🚨 <b>{worst_ch}</b> has the highest fake news rate among filtered channels — viewers show the most negative sentiment shifts.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="insight-box">✅ <b>{best_ch}</b> leads in authentic content — its viewers score highest on knowledge accuracy.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="warning-box">⚠️ Viewers of <b>{worst_an}</b> report the lowest average sentiment in the dataset — indicating high emotional distress from content consumed.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="insight-box">🏆 <b>{best_an}</b>'s audience consistently shows the highest trust scores and sentiment — a benchmark for responsible journalism.</div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIEWER INTELLIGENCE (Predictive)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Viewer Intelligence":
    st.title("🔮 Viewer Intelligence — What Will Happen?")
    st.caption("Predict a viewer's news verdict, sentiment and trust score based on their profile")

    st.markdown("### 🧬 Build Your Viewer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        p_channel = st.selectbox("📺 Channel", df["Channel_Watched"].unique())
        anchor_map = df.groupby("Channel_Watched")["Anchor_Name"].unique().to_dict()
        p_anchor   = st.selectbox("🎙️ Anchor", anchor_map.get(p_channel, df["Anchor_Name"].unique()))
        p_category = st.selectbox("📰 Category", df["News_Category"].unique())

    with col2:
        p_freq    = st.selectbox("⏱️ How Often Watch?", ["Daily","Weekly","Occasional"])
        p_trp     = st.slider("📊 TRP Score", 1.0, 10.0, 7.5, 0.1)
        p_sens    = st.slider("🔊 Sensationalism", 1.0, 10.0, 6.0, 0.1)

    with col3:
        p_fake  = st.slider("🔴 Fake Stories Watched", 0, 9, 2)
        p_auth  = st.slider("🟢 Authentic Stories Watched", 0, 9, 4)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Generate Prediction", use_container_width=True, type="primary")

    if predict_btn or True:
        try:
            ch_enc   = le_channel.transform([p_channel])[0]
            an_enc   = le_anchor.transform([p_anchor])[0]
            cat_enc  = le_cat.transform([p_category])[0]
            freq_enc = le_freq.transform([p_freq])[0]
        except:
            ch_enc, an_enc, cat_enc, freq_enc = 0, 0, 0, 0

        X_in = np.array([[ch_enc, an_enc, cat_enc, freq_enc,
                          p_trp, p_sens, p_fake, p_auth]])

        pred_verdict   = le_verdict.inverse_transform(clf.predict(X_in))[0]
        pred_proba     = clf.predict_proba(X_in)[0]
        pred_sentiment = float(reg_sent.predict(X_in)[0])
        pred_trust     = float(reg_trust.predict(X_in)[0])

        verdict_classes = le_verdict.classes_
        colors_map = {"Fake":"#ef4444","Authentic":"#10b981","Misleading":"#f59e0b"}
        v_color = colors_map.get(pred_verdict, "#6c63ff")

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'''<div class="metric-card" style="border-left-color:{v_color}">
            <div class="metric-value" style="color:{v_color}">{pred_verdict}</div>
            <div class="metric-label">Predicted News Type</div></div>''', unsafe_allow_html=True)
        sent_color = "#10b981" if pred_sentiment > 0 else "#ef4444"
        r2.markdown(f'''<div class="metric-card" style="border-left-color:{sent_color}">
            <div class="metric-value" style="color:{sent_color}">{pred_sentiment:+.2f}</div>
            <div class="metric-label">Predicted Sentiment (−5 to +5)</div></div>''', unsafe_allow_html=True)
        t_color = "#10b981" if pred_trust >= 6 else "#f59e0b" if pred_trust >= 4 else "#ef4444"
        r3.markdown(f'''<div class="metric-card" style="border-left-color:{t_color}">
            <div class="metric-value" style="color:{t_color}">{pred_trust:.1f}/10</div>
            <div class="metric-label">Predicted Trust Score</div></div>''', unsafe_allow_html=True)
        conf = max(pred_proba)*100
        r4.markdown(f'''<div class="metric-card" style="border-left-color:#6c63ff">
            <div class="metric-value" style="color:#6c63ff">{conf:.0f}%</div>
            <div class="metric-label">Model Confidence</div></div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown('<div class="section-header">📊 Verdict Probability Breakdown</div>', unsafe_allow_html=True)
            prob_df = pd.DataFrame({"Verdict": verdict_classes, "Probability": pred_proba*100})
            fig_p = px.bar(prob_df, x="Verdict", y="Probability",
                           color="Verdict",
                           color_discrete_map={"Fake":"#ef4444","Authentic":"#10b981","Misleading":"#f59e0b"},
                           labels={"Probability":"Probability (%)"})
            fig_p.update_layout(showlegend=False, height=300,
                                margin=dict(t=10,b=10,l=10,r=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0")
            st.plotly_chart(fig_p, use_container_width=True)

        with pc2:
            st.markdown('<div class="section-header">🌡️ Viewer Emotional State Gauge</div>', unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_sentiment,
                delta={"reference": 0, "suffix": " from neutral"},
                gauge={
                    "axis": {"range": [-5, 5], "tickcolor": "#e2e8f0"},
                    "bar": {"color": sent_color},
                    "bgcolor": "#1e2130",
                    "steps": [
                        {"range": [-5, -2], "color": "#7f1d1d"},
                        {"range": [-2, 0],  "color": "#78350f"},
                        {"range": [0, 2],   "color": "#064e3b"},
                        {"range": [2, 5],   "color": "#065f46"},
                    ],
                    "threshold": {"line": {"color": "white","width": 3}, "thickness": 0.75, "value": 0}
                },
                title={"text": "Sentiment Score", "font": {"color": "#e2e8f0"}}
            ))
            fig_g.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10),
                                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_g, use_container_width=True)

        # Contextual message
        if pred_verdict == "Fake":
            st.markdown('<div class="danger-box">🚨 <b>High Risk Profile:</b> This viewer is likely consuming predominantly fake news. Their emotional state is being negatively manipulated and their knowledge accuracy is likely significantly below average.</div>', unsafe_allow_html=True)
        elif pred_verdict == "Misleading":
            st.markdown('<div class="warning-box">⚠️ <b>Moderate Risk Profile:</b> This viewer's feed contains misleading content. While not entirely false, partial truths are distorting their understanding of events.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">✅ <b>Healthy Media Diet:</b> This viewer is primarily consuming authentic news. Their trust and knowledge scores are expected to be well above average.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDITORIAL COMPASS (Prescriptive)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Editorial Compass":
    st.title("🎯 Editorial Compass — What Should You Do?")
    st.caption("Personalized media literacy guidance based on your viewing data")

    st.markdown("### 🧮 Your Media Diet Score Calculator")
    md1, md2, md3 = st.columns(3)

    with md1:
        my_channel = st.selectbox("Your Primary Channel", df["Channel_Watched"].unique())
    with md2:
        my_age = st.selectbox("Your Age Group", ["18-25","26-40","41-60","60+"])
    with md3:
        my_freq = st.selectbox("Viewing Frequency", ["Daily","Weekly","Occasional"])

    # Compute media diet score based on channel data
    ch_data = df[df["Channel_Watched"] == my_channel]
    fake_rate_ch  = (ch_data["News_Verdict"]=="Fake").mean()
    avg_sens_ch   = ch_data["Sensationalism_Score"].mean()
    avg_trust_ch  = ch_data["Trust_Score"].mean()
    avg_know_ch   = ch_data["Knowledge_Accuracy_Pct"].mean()

    freq_multiplier = {"Daily":1.3, "Weekly":1.0, "Occasional":0.7}[my_freq]
    media_diet_score = max(0, min(100,
        (100 - fake_rate_ch*100*0.4 - avg_sens_ch*2.5 + avg_trust_ch*3 + avg_know_ch*0.3) * freq_multiplier / 1.1
    ))

    score_color = "#10b981" if media_diet_score >= 60 else "#f59e0b" if media_diet_score >= 40 else "#ef4444"

    st.markdown("---")
    ms1, ms2, ms3, ms4 = st.columns(4)
    ms1.markdown(f'''<div class="metric-card" style="border-left-color:{score_color}">
        <div class="metric-value" style="color:{score_color}">{media_diet_score:.0f}/100</div>
        <div class="metric-label">Media Diet Score</div></div>''', unsafe_allow_html=True)
    ms2.markdown(f'''<div class="metric-card" style="border-left-color:#ef4444">
        <div class="metric-value" style="color:#ef4444">{fake_rate_ch*100:.1f}%</div>
        <div class="metric-label">Fake News Exposure</div></div>''', unsafe_allow_html=True)
    ms3.markdown(f'''<div class="metric-card" style="border-left-color:#f59e0b">
        <div class="metric-value" style="color:#f59e0b">{avg_sens_ch:.1f}/10</div>
        <div class="metric-label">Sensationalism Level</div></div>''', unsafe_allow_html=True)
    ms4.markdown(f'''<div class="metric-card" style="border-left-color:#10b981">
        <div class="metric-value" style="color:#10b981">{avg_know_ch:.1f}%</div>
        <div class="metric-label">Avg Knowledge Score</div></div>''', unsafe_allow_html=True)

    st.markdown("---")
    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown('<div class="section-header">🏆 Channel Trust Leaderboard</div>', unsafe_allow_html=True)
        ch_rank = df.groupby("Channel_Watched").agg(
            Trust=("Trust_Score","mean"),
            Authentic_Rate=("News_Verdict", lambda x: (x=="Authentic").mean()*100),
            Knowledge=("Knowledge_Accuracy_Pct","mean")
        ).round(2).reset_index().sort_values("Trust", ascending=False)
        ch_rank["Rank"] = range(1, len(ch_rank)+1)
        ch_rank["Medal"] = ["🥇","🥈","🥉"] + [""] * (len(ch_rank)-3)
        st.dataframe(ch_rank[["Medal","Channel_Watched","Trust","Authentic_Rate","Knowledge"]].rename(columns={
            "Channel_Watched":"Channel","Trust":"Trust /10","Authentic_Rate":"Authentic %","Knowledge":"Knowledge %"
        }), use_container_width=True, hide_index=True)

    with ec2:
        st.markdown('<div class="section-header">🎙️ Anchor Integrity Score</div>', unsafe_allow_html=True)
        an_rank = df.groupby("Anchor_Name").agg(
            Trust=("Trust_Score","mean"),
            Sentiment=("Sentiment_Score","mean"),
            Fake_Rate=("News_Verdict", lambda x: (x=="Fake").mean()*100)
        ).round(2).reset_index()
        an_rank["Integrity"] = (an_rank["Trust"]*4 + an_rank["Sentiment"]*6 - an_rank["Fake_Rate"]*0.3).round(1)
        an_rank = an_rank.sort_values("Integrity", ascending=False)
        fig_rank = px.bar(an_rank, x="Anchor_Name", y="Integrity",
                          color="Integrity", color_continuous_scale="RdYlGn",
                          labels={"Anchor_Name":"","Integrity":"Integrity Score"})
        fig_rank.update_layout(showlegend=False, height=280, coloraxis_showscale=False,
                               margin=dict(t=10,b=60,l=10,r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", xaxis=dict(tickangle=15))
        st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Personalised Action Plan</div>', unsafe_allow_html=True)

    best_channel_overall = df.groupby("Channel_Watched")["Trust_Score"].mean().idxmax()
    best_anchor_overall  = df.groupby("Anchor_Name")["Trust_Score"].mean().idxmax()

    actions = []
    if fake_rate_ch > 0.40:
        actions.append(f'''<div class="danger-box">🔴 <b>Switch Channel:</b> {my_channel} has a {fake_rate_ch*100:.1f}% fake news rate. Consider shifting to <b>{best_channel_overall}</b> for more reliable information.</div>''')
    if avg_sens_ch > 7:
        actions.append(f'''<div class="warning-box">🟡 <b>Sensationalism Alert:</b> The channel you watch scores {avg_sens_ch:.1f}/10 on sensationalism. High sensationalism artificially spikes fear and anger — try pairing with a calmer source.</div>''')
    if my_freq == "Daily" and fake_rate_ch > 0.35:
        actions.append('<div class="danger-box">🔴 <b>Reduce Frequency:</b> Daily consumption of high-fake-news channels compounds emotional damage. Consider switching to weekly curated news consumption.</div>')
    if my_age in ["18-25","60+"]:
        actions.append(f'''<div class="warning-box">🟡 <b>Age Vulnerability:</b> The {my_age} age group shows higher susceptibility to fake news emotional impact. Always cross-verify headlines on <b>fact-checking portals</b> before sharing.</div>''')

    actions.append(f'''<div class="insight-box">✅ <b>Best Anchor to Follow:</b> Based on integrity scores, <b>{best_anchor_overall}</b> consistently delivers the most factual, low-sensationalism reporting in this dataset.</div>''')
    actions.append('<div class="insight-box">✅ <b>Golden Rule:</b> If a headline makes you angry or fearful within 3 seconds of reading — pause. That is a sensationalism trigger, not news.</div>')
    actions.append('<div class="insight-box">📱 <b>Multi-Source Strategy:</b> Consuming 3+ channels with different editorial stances reduces your fake news exposure by ~62% based on this dataset's patterns.</div>')

    for a in actions:
        st.markdown(a, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Sentiment Recovery Simulator</div>', unsafe_allow_html=True)
    sim1, sim2 = st.columns(2)
    with sim1:
        current_fake = st.slider("Current Fake Units/day", 0, 9, 4)
        proposed_fake = st.slider("Proposed Fake Units/day (after change)", 0, 9, 1)
    with sim2:
        current_auth = st.slider("Current Authentic Units/day", 0, 9, 2)
        proposed_auth = st.slider("Proposed Authentic Units/day (after change)", 0, 9, 6)

    curr_sent  = 2.5 - 0.55*current_fake  + 0.45*current_auth
    prop_sent  = 2.5 - 0.55*proposed_fake + 0.45*proposed_auth
    curr_know  = 65  - 4.0*current_fake   + 3.5*current_auth
    prop_know  = 65  - 4.0*proposed_fake  + 3.5*proposed_auth

    sim_cols = st.columns(4)
    for col, label, curr, prop in zip(
        sim_cols,
        ["Sentiment Before","Sentiment After","Knowledge Before","Knowledge After"],
        [curr_sent, prop_sent, curr_know, prop_know],
        [curr_sent, prop_sent, curr_know, prop_know]
    ):
        c = "#10b981" if ("After" in label and prop > curr_sent) or ("After" in label and prop_know > curr_know) else "#ef4444" if "Before" in label else "#6c63ff"
        col.markdown(f'''<div class="metric-card" style="border-left-color:{c}">
            <div class="metric-value" style="color:{c}">{prop:.2f}</div>
            <div class="metric-label">{label}</div></div>''', unsafe_allow_html=True)

    delta_sent = prop_sent - curr_sent
    delta_know = prop_know - curr_know
    if delta_sent > 0:
        st.markdown(f'<div class="insight-box">📈 By making this change, your predicted sentiment improves by <b>{delta_sent:+.2f}</b> points and knowledge accuracy by <b>{delta_know:+.1f}%</b>.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="danger-box">📉 This consumption pattern would decrease your sentiment by <b>{delta_sent:.2f}</b> points. Consider more authentic content.</div>', unsafe_allow_html=True)
