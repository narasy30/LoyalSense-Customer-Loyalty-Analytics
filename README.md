# LoyalSense — Customer Loyalty Analytics

A real-time data analytics project. A customer base is profiled with RFM
(Recency, Frequency, Monetary), segmented **without any labels** using K-Means,
and scored for churn risk and lifetime value. A live stream of customer events
then arrives one at a time — and because the system holds **one live profile per
customer**, it can detect the exact moment somebody moves between segments and
fire the right loyalty action.

Read `LoyalSense_Customer_Loyalty_Analytics.pdf` first. It contains the full
requirement specification, the architecture, and a line-by-line explanation of
every file in `src/`.

---

## 1. Quick start (3 commands)

```bash
pip install -r requirements.txt
cd src
python main.py
```

That is the whole project. It runs all five stages and writes every artefact
into `outputs/`.

| Command | What it does |
|---|---|
| `python main.py` | Full run, 1 second pause between events |
| `python main.py --fast` | Same, no pause (good while debugging) |
| `python main.py --no-charts` | Console output only, skips matplotlib |

Each module also runs standalone:

```bash
python data_loader.py       # ingestion + RFM derivation
python rfm_engine.py        # scores, segments, churn risk, CLV
python eda.py               # regenerate the four charts
python model.py             # re-run the clustering
python stream_simulator.py  # replay the live event stream
```

---

## 2. Folder structure

```
LoyalSense/
├── README.md
├── requirements.txt
├── LoyalSense_Customer_Loyalty_Analytics.pdf   <- the handbook, start here
├── data/
│   ├── customers.csv        20 customers with their purchase history
│   └── live_events.csv      16 events for the real-time feed
├── src/
│   ├── config.py            paths, segment grid, churn rules, action matrix
│   ├── data_loader.py       ingestion, validation, RFM derivation
│   ├── rfm_engine.py        scoring, segments, churn risk, CLV, actions
│   ├── eda.py               console statistics + 4 charts
│   ├── model.py             K-Means, elbow + silhouette, cluster naming
│   ├── stream_simulator.py  stateful real-time loop with migration detection
│   └── main.py              orchestrator (run this)
├── app/
│   └── dashboard.py         OPTIONAL Streamlit control room
├── sample_outputs/          reference results so you can check your run
└── outputs/                 empty until you run the project
```

---

## 3. What gets generated in `outputs/`

| File | Contents |
|---|---|
| `01_segment_revenue.png` | Revenue and customer count per segment |
| `02_rfm_map.png` | Recency vs monetary, bubble = frequency |
| `03_churn_vs_value.png` | Risk against value, with the escalation zone |
| `04_segment_profiles.png` | Average R, F and M score per segment |
| `segment_model.joblib` | Fitted scaler + K-Means and the cluster names |
| `scored_customers.csv` | All 20 customers with every derived field |
| `realtime_actions.csv` | One audit row per streamed event |
| `summary_report.txt` | Plain-text run summary |

`sample_outputs/` holds the exact files a correct run produces, plus
`console_output.txt` with the full reference log. Compare your `outputs/`
against it — only the timestamps should differ.

---

## 4. Optional dashboard

```bash
pip install streamlit
python src/main.py          # must run once so the model file exists
streamlit run app/dashboard.py
```

Three tabs: a live event stream with animated KPI tiles and migration banners,
a customer-base explorer, and a single-customer view with a purchase simulator.

---

## 5. Requirements

Python 3.9 or newer. Dependencies are in `requirements.txt`
(pandas, numpy, scikit-learn, matplotlib, joblib). No internet access,
no database and no API key is required.
