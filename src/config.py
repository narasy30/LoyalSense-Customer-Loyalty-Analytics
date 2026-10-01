"""
config.py
---------
One place for every path, threshold, segment rule and loyalty action used by
the project. If it is a number or a policy a marketing manager could argue
about, it lives here and nowhere else.
"""

from datetime import date
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. FOLDER LAYOUT
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"
EVENTS_CSV = DATA_DIR / "live_events.csv"

MODEL_FILE = OUTPUT_DIR / "segment_model.joblib"
SCORED_CUSTOMERS = OUTPUT_DIR / "scored_customers.csv"
EVENT_LOG = OUTPUT_DIR / "realtime_actions.csv"
SUMMARY_REPORT = OUTPUT_DIR / "summary_report.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. ANALYSIS DATE
# ----------------------------------------------------------------------------
# Recency is measured against this date. Hard-coding it keeps the whole project
# reproducible; a live system would use date.today().
AS_OF_DATE = date(2026, 4, 1)

# ----------------------------------------------------------------------------
# 3. RFM SCORING
# ----------------------------------------------------------------------------
RFM_BINS = 5              # quintiles -> scores 1 to 5

# Features handed to the clustering model. Note there is NO target column
# anywhere in this project: segmentation is unsupervised.
CLUSTER_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "return_rate",
    "tenure_days",
]

K_RANGE = range(2, 7)     # candidate cluster counts to evaluate
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 4. SEGMENT RULE GRID
# ----------------------------------------------------------------------------
# Ordered list; the first rule whose ranges all match wins.
# (segment name, r_min, r_max, f_min, f_max, m_min, m_max)
SEGMENT_RULES = [
    ("Champions",          4, 5, 4, 5, 4, 5),
    ("Cannot Lose Them",   1, 2, 4, 5, 4, 5),
    ("Loyal Customers",    3, 5, 3, 5, 3, 5),
    ("At Risk",            1, 2, 3, 5, 1, 5),
    ("Potential Loyalist", 4, 5, 2, 3, 1, 5),
    ("New Customers",      4, 5, 1, 1, 1, 5),
    ("Promising",          3, 5, 1, 2, 1, 5),
    ("Needs Attention",    3, 3, 3, 3, 1, 5),
    ("Hibernating",        2, 2, 1, 2, 1, 5),
    ("Lost",               1, 1, 1, 2, 1, 5),
    ("Others",             1, 5, 1, 5, 1, 5),   # catch-all, must stay last
]

# ----------------------------------------------------------------------------
# 5. CHURN RISK RULES (additive penalties, capped at 100)
# ----------------------------------------------------------------------------
RECENCY_BANDS = [          # (max days since last purchase, penalty)
    (30, 0),
    (60, 8),
    (120, 20),
    (240, 35),
    (10_000, 50),
]

RETURN_RATE_BANDS = [      # (max return rate, penalty)
    (0.05, 0),
    (0.10, 6),
    (0.18, 14),
    (1.00, 22),
]

RATING_BANDS = [           # (minimum average rating, penalty)
    (4.5, 0),
    (4.0, 5),
    (3.5, 12),
    (0.0, 20),
]

# How many past ratings a new review is averaged against. Order count is a
# poor proxy: a customer with 74 orders has not left 74 reviews, and using that
# weight would make a fresh 1-star review mathematically invisible.
RATING_MEMORY = 5

TICKET_PENALTY = 4          # points per open support ticket
TICKET_PENALTY_CAP = 20

CHURN_BANDS = [            # (max score, label)
    (25, "LOW"),
    (50, "MEDIUM"),
    (100, "HIGH"),
]

# ----------------------------------------------------------------------------
# 6. CUSTOMER LIFETIME VALUE
# ----------------------------------------------------------------------------
GROSS_MARGIN = 0.32         # share of revenue that is actually profit
MAX_HORIZON_YEARS = 3.0     # how far ahead the business is willing to project

# ----------------------------------------------------------------------------
# 7. LOYALTY ACTION MATRIX
# ----------------------------------------------------------------------------
# segment -> (action, offer, loyalty-point multiplier on a purchase)
ACTION_MATRIX = {
    "Champions":          ("Invite to the VIP tier and early-access drops",
                           "Free express delivery for 12 months", 3.0),
    "Cannot Lose Them":   ("Personal win-back call from an account manager",
                           "25% voucher, valid 14 days", 3.0),
    "Loyal Customers":    ("Cross-sell from the adjacent category",
                           "Bonus points on the next 3 orders", 2.0),
    "At Risk":            ("Automated win-back email sequence",
                           "15% voucher, valid 21 days", 2.0),
    "Potential Loyalist": ("Nudge towards a subscription plan",
                           "Free trial month", 1.5),
    "New Customers":      ("Onboarding journey and product education",
                           "10% off the second order", 1.5),
    "Promising":          ("Encourage a second category purchase",
                           "Bundle discount", 1.5),
    "Needs Attention":    ("Re-engagement offer before recency worsens",
                           "10% voucher, valid 30 days", 1.0),
    "Hibernating":        ("Low-cost reactivation email",
                           "Free delivery on any order", 1.0),
    "Lost":               ("Move to a quarterly newsletter only",
                           "No offer - not economical", 1.0),
    "Others":             ("Review manually", "No offer", 1.0),
}

# How "good" each segment is. Used only to decide whether a customer moving
# between segments has been upgraded or downgraded.
SEGMENT_RANK = {
    "Champions": 10,
    "Loyal Customers": 9,
    "Cannot Lose Them": 8,
    "Potential Loyalist": 7,
    "Promising": 6,
    "New Customers": 5,
    "Needs Attention": 4,
    "At Risk": 3,
    "Hibernating": 2,
    "Lost": 1,
    "Others": 0,
}

# Escalations applied on top of the segment action
# A personal call from a human costs money, so escalate only when the projected
# margin justifies it. CLV here is gross margin over at most three years.
HIGH_VALUE_CLV = 15_000         # rupees of projected margin
ESCALATE_ACTION = "Escalate to the retention desk within 24 hours"

BASE_POINTS_PER_RUPEE = 0.02    # 2 points per rupee spent, before multipliers

# ----------------------------------------------------------------------------
# 8. STREAM SETTINGS
# ----------------------------------------------------------------------------
STREAM_DELAY_SECONDS = 1.0
