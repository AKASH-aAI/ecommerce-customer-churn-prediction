import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_customers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")

app = Flask(__name__)
app.secret_key = "customeriq-secret-key-change-in-production"

# ---- Model inputs (exact 24 features) ----
FEATURE_COLUMNS = [
    "Age", "Gender", "Country", "City", "Membership_Years",
    "Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
    "Cart_Abandonment_Rate", "Wishlist_Items", "Total_Purchases",
    "Average_Order_Value", "Days_Since_Last_Purchase",
    "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
    "Customer_Service_Calls", "Product_Reviews_Written",
    "Social_Media_Engagement_Score", "Mobile_App_Usage",
    "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance",
    "Signup_Quarter",
]

NUMERIC_FEATURES = {
    "Age", "Membership_Years", "Login_Frequency", "Session_Duration_Avg",
    "Pages_Per_Session", "Cart_Abandonment_Rate", "Wishlist_Items",
    "Total_Purchases", "Average_Order_Value", "Days_Since_Last_Purchase",
    "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
    "Customer_Service_Calls", "Product_Reviews_Written",
    "Social_Media_Engagement_Score", "Mobile_App_Usage",
    "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance",
}

CATEGORICAL_FEATURES = {
    "Gender", "Country", "City", "Signup_Quarter",
}

# ---- Cached loaders ----
_model = None
_threshold = None
_df = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def load_threshold():
    global _threshold
    if _threshold is None:
        if not os.path.exists(THRESHOLD_PATH):
            raise FileNotFoundError(f"Threshold file not found at {THRESHOLD_PATH}")
        _threshold = joblib.load(THRESHOLD_PATH)
        # threshold.pkl may be a scalar, list, or ndarray
        if hasattr(_threshold, "item") and not isinstance(_threshold, (list, tuple)):
            try:
                _threshold = _threshold.item()
            except Exception:
                pass
        if isinstance(_threshold, (list, tuple, np.ndarray)):
            _threshold = float(np.array(_threshold).ravel()[0])
        _threshold = float(_threshold)
    return _threshold


def load_df():
    global _df
    if _df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        _df = pd.read_csv(DATA_PATH)
    return _df


def predict_probability(feature_row_df):
    """Given a DataFrame with the 24 feature columns, return churn probability."""
    model = load_model()
    # Ensure column order matches training
    feature_row_df = feature_row_df[FEATURE_COLUMNS]
    proba = model.predict_proba(feature_row_df)[0][1]
    return float(proba)


def risk_label(prob, thr):
    return "High Risk" if prob >= thr else "Low Risk"


# ---------------- Routes ----------------

@app.route("/")
def index():
    df = load_df()
    thr = load_threshold()

    total_customers = int(len(df))
    churned = int(df["Churned"].sum()) if "Churned" in df.columns else 0
    active = total_customers - churned
    churn_rate = (churned / total_customers * 100) if total_customers else 0.0
    avg_ltv = float(df["Lifetime_Value"].mean()) if "Lifetime_Value" in df.columns else 0.0

    # High-risk: score every customer with the model (may be slow on 50k).
    # To keep dashboard responsive we sample if the dataset is large.
    high_risk_count = None
    high_risk_table = []
    try:
        model = load_model()
        X = df[FEATURE_COLUMNS].copy()
        # Batch predict in chunks for performance
        probs = model.predict_proba(X)[:, 1]
        df_scored = df[["CustomerID"] + ["Age", "Country", "Lifetime_Value", "Total_Purchases"]].copy()
        df_scored["ChurnProb"] = probs
        df_scored["Risk"] = np.where(probs >= thr, "High Risk", "Low Risk")
        high_risk_count = int((probs >= thr).sum())
        high_risk_table = (
            df_scored.sort_values("ChurnProb", ascending=False)
            .head(12)
            .to_dict(orient="records")
        )
    except Exception:
        high_risk_count = None

    kpis = {
        "total_customers": total_customers,
        "churned": churned,
        "active": active,
        "churn_rate": round(churn_rate, 2),
        "high_risk": high_risk_count,
        "avg_ltv": round(avg_ltv, 2),
    }
    return render_template("index.html", kpis=kpis, threshold=thr,
                           high_risk_table=high_risk_table, page="overview")


@app.route("/search", methods=["GET", "POST"])
def search():
    df = load_df()
    thr = load_threshold()
    query = ""
    if request.method == "POST":
        query = request.form.get("customer_id", "").strip().upper()
        if not query:
            flash("Please enter a Customer ID.", "warning")
            return render_template("search.html", query=query, page="search")

        match = df[df["CustomerID"].astype(str).str.upper() == query]
        if match.empty:
            flash(f"Customer '{query}' not found.", "danger")
            return render_template("search.html", query=query, page="search")

        row = match.iloc[0]
        input_row = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()])
        try:
            prob = predict_probability(input_row)
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")
            return render_template("search.html", query=query, page="search")

        result = {
            "customer_id": row["CustomerID"],
            "probability": round(prob * 100, 2),
            "risk": risk_label(prob, thr),
            "threshold": thr,
        }
        # Detail sections (exclude Churned on purpose)
        demographics = {
            "Age": row.get("Age"), "Gender": row.get("Gender"),
            "Country": row.get("Country"), "City": row.get("City"),
        }
        membership = {
            "Membership_Years": row.get("Membership_Years"),
            "Signup_Quarter": row.get("Signup_Quarter"),
            "Lifetime_Value": row.get("Lifetime_Value"),
            "Credit_Balance": row.get("Credit_Balance"),
        }
        behavior = {
            "Total_Purchases": row.get("Total_Purchases"),
            "Average_Order_Value": row.get("Average_Order_Value"),
            "Days_Since_Last_Purchase": row.get("Days_Since_Last_Purchase"),
            "Cart_Abandonment_Rate": row.get("Cart_Abandonment_Rate"),
            "Discount_Usage_Rate": row.get("Discount_Usage_Rate"),
            "Returns_Rate": row.get("Returns_Rate"),
        }
        engagement = {
            "Login_Frequency": row.get("Login_Frequency"),
            "Session_Duration_Avg": row.get("Session_Duration_Avg"),
            "Pages_Per_Session": row.get("Pages_Per_Session"),
            "Email_Open_Rate": row.get("Email_Open_Rate"),
            "Customer_Service_Calls": row.get("Customer_Service_Calls"),
            "Product_Reviews_Written": row.get("Product_Reviews_Written"),
            "Social_Media_Engagement_Score": row.get("Social_Media_Engagement_Score"),
            "Mobile_App_Usage": row.get("Mobile_App_Usage"),
            "Payment_Method_Diversity": row.get("Payment_Method_Diversity"),
        }
        return render_template("result.html", result=result, demographics=demographics,
                               membership=membership, behavior=behavior,
                               engagement=engagement, page="search")

    return render_template("search.html", query=query, page="search")


@app.route("/manual", methods=["GET"])
def manual():
    thr = load_threshold()
    return render_template("manual.html", threshold=thr, page="manual")


@app.route("/manual-predict", methods=["POST"])
def manual_predict():
    thr = load_threshold()
    raw = {}
    provided = 0
    blank = 0

    for col in FEATURE_COLUMNS:
        val = request.form.get(col, "").strip()
        if val == "":
            raw[col] = np.nan
            blank += 1
            continue
        provided += 1
        if col in NUMERIC_FEATURES:
            try:
                raw[col] = float(val)
            except ValueError:
                flash(f"'{col}' must be numeric.", "danger")
                return redirect(url_for("manual"))
        else:
            raw[col] = val

    if provided == 0:
        flash("Please fill at least one field before predicting.", "warning")
        return redirect(url_for("manual"))

    input_df = pd.DataFrame([raw])
    try:
        prob = predict_probability(input_df)
    except Exception as e:
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("manual"))

    # Business interpretation
    if prob >= thr:
        interpretation = (
            "This customer's behavior pattern is similar to previously churned customers. "
            "Consider proactive retention offers, personalized outreach, or re-engagement campaigns."
        )
    else:
        interpretation = (
            "This customer's behavior pattern indicates low churn likelihood. "
            "Continue standard engagement to maintain loyalty."
        )

    result = {
        "probability": round(prob * 100, 2),
        "risk": risk_label(prob, thr),
        "threshold": thr,
        "provided": provided,
        "blank": blank,
        "interpretation": interpretation,
    }
    customer_info = {
        k: v for k, v in raw.items()
        if not (isinstance(v, float) and np.isnan(v))
    }
    return render_template("manual_result.html", result=result,
                           customer_info=customer_info, page="manual")


@app.route("/analytics")
def analytics():
    df = load_df().copy()
    charts = {}

    # 1. Churn Distribution – Donut
    churn_counts = df["Churned"].value_counts().reindex([0, 1], fill_value=0)
    charts["churn_dist"] = {
        "labels": ["Not Churned", "Churned"],
        "values": [int(churn_counts.get(0, 0)), int(churn_counts.get(1, 0))],
    }

    # Helper
    def group_rate(col, bins=None, labels=None, ascending=True):
        d = df.copy()
        if bins is not None:
            d["_g"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
        else:
            d["_g"] = d[col]
        agg = d.groupby("_g", observed=True)["Churned"].agg(["mean", "count"]).reset_index()
        agg["rate"] = (agg["mean"] * 100).round(2)
        if ascending:
            agg = agg.sort_values("_g")
        return {"x": [str(v) for v in agg["_g"].tolist()],
                "y": agg["rate"].tolist(),
                "counts": agg["count"].tolist()}

    # 2. Churn by Gender
    charts["gender"] = group_rate("Gender")

    # 3. Churn by Country (horizontal top 10)
    country = (
        df.groupby("Country")["Churned"].agg(["mean", "count"]).reset_index()
        .sort_values("count", ascending=False).head(10)
    )
    country["rate"] = (country["mean"] * 100).round(2)
    charts["country"] = {
        "x": country["rate"].tolist(),
        "y": country["Country"].astype(str).tolist(),
        "counts": country["count"].tolist(),
    }

    # 4. Churn by Age Group
    charts["age_group"] = group_rate(
        "Age", bins=[0, 25, 35, 45, 55, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
    )

    # 5. Churn by Membership Years (line)
    mem = (
        df.groupby("Membership_Years")["Churned"].agg(["mean", "count"]).reset_index()
        .sort_values("Membership_Years")
    )
    charts["membership"] = {
        "x": mem["Membership_Years"].astype(float).round(2).tolist(),
        "y": (mem["mean"] * 100).round(2).tolist(),
        "counts": mem["count"].tolist(),
    }

    # 6. Login Frequency vs Churn (bucketed)
    charts["login"] = group_rate(
        "Login_Frequency",
        bins=[-0.1, 1, 3, 6, 10, 100],
        labels=["≤1", "2-3", "4-6", "7-10", "10+"],
    )

    # 7. Cart Abandonment vs Churn
    charts["cart"] = group_rate(
        "Cart_Abandonment_Rate",
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )

    # 8. LTV Distribution – Histogram
    ltv = df["Lifetime_Value"].dropna()
    counts, bin_edges = np.histogram(ltv, bins=30)
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    charts["ltv"] = {
        "x": [round(c, 2) for c in centers],
        "y": counts.tolist(),
    }

    # 9. Days Since Last Purchase vs Churn
    charts["days_since"] = group_rate(
        "Days_Since_Last_Purchase",
        bins=[-1, 30, 60, 90, 180, 365, 10000],
        labels=["≤30", "31-60", "61-90", "91-180", "181-365", "365+"],
    )

    # 10. Churn by Signup Quarter
    charts["quarter"] = group_rate("Signup_Quarter")

    return render_template("analytics.html", charts=charts, page="analytics")


@app.route("/segments")
def segments():
    df = load_df().copy()
    ltv_q75 = df["Lifetime_Value"].quantile(0.75)
    purch_q75 = df["Total_Purchases"].quantile(0.75)
    login_q25 = df["Login_Frequency"].quantile(0.25)
    dslp_q75 = df["Days_Since_Last_Purchase"].quantile(0.75)

    def assign(row):
        if row["Lifetime_Value"] >= ltv_q75 and row["Total_Purchases"] >= purch_q75:
            return "High Value Customers"
        if row["Membership_Years"] >= 3 and row["Total_Purchases"] >= purch_q75:
            return "Loyal Customers"
        if row["Days_Since_Last_Purchase"] >= dslp_q75:
            return "At-Risk Customers"
        if row["Login_Frequency"] <= login_q25:
            return "Low Engagement Customers"
        if row["Total_Purchases"] >= purch_q75:
            return "Frequent Buyers"
        return "Regular Customers"

    df["Segment"] = df.apply(assign, axis=1)
    seg_counts = df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    seg_churn = (
        df.groupby("Segment")["Churned"].mean().mul(100).round(2).reset_index()
    )
    seg_churn.columns = ["Segment", "ChurnRate"]

    segment_data = seg_counts.merge(seg_churn, on="Segment").to_dict(orient="records")

    rules = {
        "High Value Customers": f"Lifetime_Value ≥ {ltv_q75:.2f} AND Total_Purchases ≥ {purch_q75:.0f}",
        "Loyal Customers": f"Membership_Years ≥ 3 AND Total_Purchases ≥ {purch_q75:.0f}",
        "At-Risk Customers": f"Days_Since_Last_Purchase ≥ {dslp_q75:.0f}",
        "Low Engagement Customers": f"Login_Frequency ≤ {login_q25:.2f}",
        "Frequent Buyers": f"Total_Purchases ≥ {purch_q75:.0f}",
        "Regular Customers": "All remaining customers",
    }

    return render_template("segments.html", segments=segment_data, rules=rules,
                           page="segments")


@app.route("/model-insights")
def model_insights():
    thr = load_threshold()
    top_features = []
    try:
        model = load_model()
        clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
        pre = model.named_steps["preprocessor"] if hasattr(model, "named_steps") else None

        if clf is not None and pre is not None and hasattr(clf, "feature_importances_"):
            names = pre.get_feature_names_out()
            importances = clf.feature_importances_
            pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:10]
            top_features = [{"name": n, "importance": float(v)} for n, v in pairs]
    except Exception:
        top_features = []

    metrics = {
        "algorithm": "Random Forest Classifier",
        "n_estimators": 200,
        "max_depth": 15,
        "train_acc": 96.51,
        "test_acc": 91.07,
        "roc_auc": 0.8794,
        "threshold": thr,
    }
    return render_template("model_insights.html", metrics=metrics,
                           top_features=top_features, page="model")


@app.route("/about")
def about():
    return render_template("about.html", page="about")


@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", page="", not_found=True), 404


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)