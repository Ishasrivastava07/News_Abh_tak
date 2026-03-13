# News_Abh_tak
# ðŸ“¡ India News Integrity Monitor

A Streamlit dashboard that analyses fake news consumption patterns across Indian news channels
and their impact on viewer sentiment, trust, and knowledge accuracy.

## ðŸš€ Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## ðŸ“ Project Structure
```
â”œâ”€â”€ app.py                                    # Main Streamlit dashboard
â”œâ”€â”€ indian_news_media_integrity_dataset.csv   # Synthetic dataset (2,500 rows)
â”œâ”€â”€ requirements.txt                          # Python dependencies
â””â”€â”€ README.md
```

## ðŸ“Š Dashboard Pages

| Page | What It Does |
|------|-------------|
| ðŸ  **Media Pulse** | Overall snapshot â€” verdict distribution, fake news rates, sentiment trends |
| ðŸ”¬ **Influence Decoder** | Root cause analysis â€” TRP vs fake news, correlation heatmaps, regional insights |
| ðŸ”® **Viewer Intelligence** | ML-powered prediction â€” enter any viewer profile, get verdict + sentiment + trust prediction |
| ðŸŽ¯ **Editorial Compass** | Action plan â€” channel leaderboard, anchor integrity scores, sentiment recovery simulator |

## ðŸ› ï¸ Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/india-news-integrity-monitor.git
cd india-news-integrity-monitor
pip install -r requirements.txt
streamlit run app.py
```

## â˜ï¸ Deploy on Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** â†’ Select your repo â†’ Set `app.py` as main file
4. Click **Deploy** â€” live in ~2 minutes!

## ðŸ§  ML Models Used
- **Random Forest Classifier** â€” Predicts News Verdict (Fake / Authentic / Misleading)
- **Gradient Boosting Regressor** â€” Predicts Sentiment Score and Trust Score

## ðŸ“Œ Dataset Variables
| Column | Description |
|--------|-------------|
| Viewer_ID | Unique viewer ID |
| Month_Year | Jan 2022 â€“ Dec 2025 |
| Age_Group | 18â€“25, 26â€“40, 41â€“60, 60+ |
| Gender | Male / Female / Other |
| Region | North / South / East / West / Central |
| Channel_Watched | 6 Indian news channels |
| Anchor_Name | 7 anchors including Arnab, Ravish, Palki, Navika, Anjana |
| News_Verdict | **Target** â€” Fake / Authentic / Misleading |
| TRP_Score | Channel TRP at time of viewing |
| Sensationalism_Score | 1â€“10 headline drama score |
| Fake_Units_Consumed | Count of fake stories watched |
| Authentic_Units_Consumed | Count of authentic stories watched |
| Sentiment_Score | âˆ’5 (fear/anger) to +5 (calm/informed) |
| Trust_Score | 1â€“10 trust in channel |
| Knowledge_Accuracy_Pct | % of news facts correctly recalled |

## ðŸŽ“ Academic Context
Built as part of a Data Analytics (MGB) individual project submission.
Business idea: Authenticating facts from Indian news channels and measuring audience sentiment impact.
