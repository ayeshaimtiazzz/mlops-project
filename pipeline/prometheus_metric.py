from prometheus_client import Counter, Histogram, Gauge, ProcessCollector, PlatformCollector

# -----------------------------
# Counters
# -----------------------------

# Total number of predictions made
PREDICTION_COUNT = Counter(
    "digit_prediction_total",
    "Total number of digit predictions made"
)

# Total number of correct predictions
CORRECT_PREDICTION_COUNT = Counter(
    "correct_digit_predictions_total",
    "Total number of correct digit predictions"
)

# Total number of predictions with data drift detected
DATA_DRIFT_COUNT = Counter(
    "digit_prediction_drift_total",
    "Total number of digit predictions where data drift was detected"
)

# -----------------------------
# Histogram
# -----------------------------

# Time taken to make each prediction (seconds)
# Custom buckets for better latency visualization
PREDICTION_LATENCY = Histogram(
    "digit_prediction_latency_seconds",
    "Time taken to make each digit prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# -----------------------------
# Gauges
# -----------------------------

# Number of active users currently using the app
ACTIVE_USERS = Gauge(
    "active_users",
    "Number of active users currently using the app"
)
