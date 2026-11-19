"""
Student Performance Analysis and Prediction
--------------------------------------------
Place this file in the root of your project:

    student-performance-predictor/
    ├── app.py                         <- this file
    ├── data/StudentPerformanceFactors.csv
    └── model/model.pkl

Run with:  streamlit run app.py
Requires:  streamlit>=1.37, pandas, numpy, plotly, scikit-learn, joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# --------------------------------------------------------------------------
# Configuration  (edit these if your training script used different values)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "StudentPerformanceFactors.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
COMPARISON_PATH = BASE_DIR / "model" / "model_comparison.csv"  # optional

TARGET = "Exam_Score"

# Must match the order the model was trained with. Do not reorder.
FEATURES = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
    "Motivation_Level", "Internet_Access", "Tutoring_Sessions", "Family_Income",
    "Teacher_Quality", "School_Type", "Peer_Influence", "Physical_Activity",
    "Learning_Disabilities", "Parental_Education_Level", "Distance_from_Home",
    "Gender",
]
NUMERIC = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity",
]
CATEGORICAL = [f for f in FEATURES if f not in NUMERIC]

# Assumed to be the same split used when the model was trained.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Colours
NAVY, BLUE, PURPLE, GREEN = "#0f1b3d", "#2f6fed", "#7c5cff", "#22c55e"

st.set_page_config(page_title="Student Performance Analytics",
                   page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = f"""
<style>
.stApp {{ background: #f5f7fb; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #cdd6f4; }}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavLink"] {{
    border-radius: 10px; margin: 2px 8px; padding: 0.55rem 0.8rem;
}}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: {BLUE}; color: #fff;
}}
[data-testid="stSidebarNav"] a[aria-current="page"] *,
[data-testid="stSidebarNavLink"][aria-current="page"] * {{ color: #fff; font-weight: 600; }}
.brand {{ font-size: 1.15rem; font-weight: 700; color: #fff; padding: .5rem 1rem 1rem; }}
.brand small {{ display:block; font-weight:400; color:#8fa0d6; font-size:.75rem; }}

/* Cards */
.card {{
    background: #fff; border-radius: 16px; padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(15,27,61,.07); height: 100%;
}}
.metric-label {{ color: #6b7280; font-size: .85rem; }}
.metric-value {{ color: {NAVY}; font-size: 2rem; font-weight: 700; line-height: 1.2; }}
.metric-sub {{ color: #9ca3af; font-size: .75rem; }}
.page-title {{ color: {NAVY}; font-weight: 800; font-size: 1.9rem; margin-bottom: .2rem; }}
.page-sub {{ color: #6b7280; margin-bottom: 1.2rem; }}

.banner {{
    background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 60%, {PURPLE} 100%);
    border-radius: 20px; padding: 2rem 2.2rem; color: #fff;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 8px 24px rgba(47,111,237,.25); margin-bottom: 1.2rem;
}}
.banner h1 {{ color:#fff; font-size: 2rem; margin:0 0 .4rem; }}
.banner p {{ color:#e0e7ff; margin:0; max-width: 560px; }}
.banner .art {{ font-size: 3.4rem; letter-spacing: .6rem; }}

.result {{
    background: linear-gradient(135deg, {BLUE}, {PURPLE});
    color:#fff; border-radius: 20px; padding: 1.6rem; text-align:center;
    box-shadow: 0 8px 24px rgba(124,92,255,.3);
}}
.result .score {{ font-size: 3.4rem; font-weight: 800; }}
.insight {{
    border-left: 4px solid {GREEN}; background:#fff; border-radius: 10px;
    padding: .7rem 1rem; margin-bottom: .6rem;
    box-shadow: 0 1px 6px rgba(15,27,61,.06);
}}

/* Primary buttons */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
    background: {BLUE}; color:#fff; border:0; border-radius: 10px;
    padding: .6rem 1.4rem; font-weight: 600;
}}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
    background: {PURPLE}; color:#fff;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers: loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def safe_load():
    """Load data + model, showing a friendly error instead of a traceback."""
    try:
        return load_data(), load_model()
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Check the `data/` and `model/` folders.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load data or model: {e}")
    st.stop()


def page_header(title: str, subtitle: str):
    st.markdown(f'<div class="page-title">{title}</div>'
                f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif", color=NAVY),
    )
    return fig


# --------------------------------------------------------------------------
# Helpers: feeding data to the model
# --------------------------------------------------------------------------
# Your model may be (a) a full sklearn Pipeline that accepts raw text columns,
# or (b) a bare Ridge model trained on encoded numbers. These strategies try
# each option in turn and use the first one that the saved model accepts.
# The model itself is never retrained or modified.

def _raw(model, df, ref):
    return df[FEATURES]


def _label_encoded(model, df, ref):
    out = df[FEATURES].copy()
    for col in CATEGORICAL:
        classes = sorted(ref[col].dropna().unique())  # same as LabelEncoder
        out[col] = out[col].map({c: i for i, c in enumerate(classes)})
    return out


def _dummies(model, df, ref):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("Model has no feature_names_in_")
    both = pd.concat([ref[FEATURES], df[FEATURES]], ignore_index=True)
    enc = pd.get_dummies(both, columns=CATEGORICAL).iloc[len(ref):]
    return enc.reindex(columns=names, fill_value=0).astype(float)


STRATEGIES = [("raw", _raw), ("label-encoded", _label_encoded), ("one-hot", _dummies)]


def build_features(model, df: pd.DataFrame, ref: pd.DataFrame):
    """Return (X, strategy_name) using the first strategy the model accepts."""
    probe = df.head(5)
    last_err = None
    for name, fn in STRATEGIES:
        try:
            model.predict(fn(model, probe, ref))
            return fn(model, df, ref), name
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Saved model did not accept any input format. Last error: {last_err}")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def home_page():
    df, _ = safe_load()
    st.markdown(
        """
        <div class="banner">
          <div>
            <h1>Student Performance Analysis and Prediction</h1>
            <p>Explore the factors behind student success and estimate examination
            scores using a trained Ridge Regression model built on real student data.</p>
          </div>
          <div class="art">🎓 📚 ✏️ 🏫</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Student Records", f"{len(df):,}", "rows in dataset")
    with c2:
        metric_card("Input Features", str(len(FEATURES)), "used by the model")
    with c3:
        metric_card("Average Examination Score", f"{df[TARGET].mean():.2f}",
                    f"range {df[TARGET].min():.0f} – {df[TARGET].max():.0f}")

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        fig = px.histogram(df, x=TARGET, nbins=30, color_discrete_sequence=[BLUE],
                           title="Examination Score Distribution")
        fig.update_layout(xaxis_title="Exam Score", yaxis_title="Number of Students",
                          bargap=0.05)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        st.markdown("#### Key Insights")
        num_corr = df[NUMERIC + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(
            ascending=False)
        top = num_corr.index[0]
        st.markdown(
            f'<div class="insight">Strongest numeric driver: <b>{top.replace("_"," ")}</b> '
            f'(correlation {num_corr.iloc[0]:.2f} with exam score).</div>',
            unsafe_allow_html=True)
        by_inv = df.groupby("Parental_Involvement")[TARGET].mean()
        st.markdown(
            f'<div class="insight">Highest average score by parental involvement: '
            f'<b>{by_inv.idxmax()}</b> ({by_inv.max():.1f}) vs '
            f'<b>{by_inv.idxmin()}</b> ({by_inv.min():.1f}).</div>',
            unsafe_allow_html=True)
        att_corr = df["Attendance"].corr(df[TARGET])
        st.markdown(
            f'<div class="insight">Attendance and exam score correlation: '
            f'<b>{att_corr:.2f}</b>.</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Go to Prediction  →"):
            st.switch_page(PAGES["Prediction"])


def dataset_page():
    df, _ = safe_load()
    page_header("Dataset", "Student Performance Factors – overview and statistics")
    st.write("This dataset records study habits, background and school-related factors "
             "for each student together with their final examination score.")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Records", f"{len(df):,}")
    with c2:
        metric_card("Total Features", str(len(FEATURES)), "excluding the target")
    with c3:
        metric_card("Target Variable", TARGET)

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Numerical Features")
        st.dataframe(pd.DataFrame({"Feature": NUMERIC + [TARGET],
                                   "Type": [str(df[c].dtype) for c in NUMERIC + [TARGET]]}),
                     use_container_width=True, hide_index=True)
    with b:
        st.markdown("#### Categorical Features")
        st.dataframe(pd.DataFrame({
            "Feature": CATEGORICAL,
            "Unique Values": [df[c].nunique() for c in CATEGORICAL],
            "Categories": [", ".join(map(str, df[c].dropna().unique())) for c in CATEGORICAL],
        }), use_container_width=True, hide_index=True)

    st.markdown("#### Missing Values")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(pd.DataFrame({"Column": miss.index, "Missing": miss.values,
                                   "Percent": (miss.values / len(df) * 100).round(2)}),
                     use_container_width=True, hide_index=True)

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df[NUMERIC + [TARGET]].describe().T.round(2), use_container_width=True)


def _scatter_with_line(df, x, title):
    fig = px.scatter(df, x=x, y=TARGET, opacity=0.4, color_discrete_sequence=[BLUE],
                     title=title)
    d = df[[x, TARGET]].dropna()
    slope, intercept = np.polyfit(d[x], d[TARGET], 1)
    xs = np.array([d[x].min(), d[x].max()])
    fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines",
                    line=dict(color=PURPLE, width=3), name="Trend")
    fig.update_layout(xaxis_title=x.replace("_", " "), yaxis_title="Exam Score")
    return style_fig(fig)


def analysis_page():
    df, _ = safe_load()
    page_header("Analysis", "How different factors relate to examination scores")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_scatter_with_line(df, "Hours_Studied",
                        "Study Hours vs Examination Score"), use_container_width=True)
    with c2:
        st.plotly_chart(_scatter_with_line(df, "Attendance",
                        "Attendance vs Examination Score"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="Sleep_Hours", y=TARGET, color_discrete_sequence=[PURPLE],
                     title="Sleep Hours vs Examination Score")
        fig.update_layout(xaxis_title="Sleep Hours", yaxis_title="Exam Score")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c4:
        order = ["Low", "Medium", "High"]
        avg = (df.groupby("Parental_Involvement")[TARGET].mean()
                 .reindex(order).dropna().reset_index())
        fig = px.bar(avg, x="Parental_Involvement", y=TARGET, text_auto=".2f",
                     color_discrete_sequence=[GREEN],
                     title="Parental Involvement vs Average Examination Score")
        fig.update_layout(xaxis_title="Parental Involvement",
                          yaxis_title="Average Exam Score")
        st.plotly_chart(style_fig(fig), use_container_width=True)


def prediction_page():
    df, model = safe_load()
    page_header("Prediction", "Enter student details to estimate the examination score")

    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(FEATURES):
            label = feat.replace("_", " ")
            with cols[i % 3]:
                if feat in NUMERIC:
                    lo, hi = int(df[feat].min()), int(df[feat].max())
                    default = int(round(df[feat].median()))
                    values[feat] = st.number_input(label, min_value=lo, max_value=hi,
                                                   value=default, step=1)
                else:
                    options = sorted(df[feat].dropna().unique().tolist())
                    values[feat] = st.selectbox(label, options)
        submitted = st.form_submit_button("Predict Examination Score")

    if not submitted:
        return

    try:
        row = pd.DataFrame([values])[FEATURES]
        X, strategy = build_features(model, row, df)
        score = float(np.ravel(model.predict(X))[0])
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        return

    st.write("")
    st.markdown(
        f'<div class="result"><div>Predicted Examination Score</div>'
        f'<div class="score">{score:.2f}</div>'
        f'<div>This is an estimate from a statistical model, not a guaranteed result.</div>'
        f'</div>', unsafe_allow_html=True)
    if score > 100:
        st.caption("The model's raw output exceeds 100; real scores cannot go above 100.")
    with st.expander("View entered information"):
        st.dataframe(row.T.rename(columns={0: "Value"}), use_container_width=True)


@st.cache_data(show_spinner="Evaluating saved model...")
def evaluate_model():
    """Compute test metrics + cross-validation from the saved model (no retraining of it)."""
    df, model = load_data(), load_model()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_test, strategy = build_features(model, test_df, df)
    y_test = test_df[TARGET]
    mask = X_test.notna().all(axis=1)           # drop rows the model cannot score
    X_test, y_test = X_test[mask], y_test[mask]
    pred = model.predict(X_test)
    results = {
        "strategy": strategy,
        "n_test": len(y_test),
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "cv": None,
    }
    try:  # cross-validation refits *clones*; the saved model is untouched
        X_all, _ = build_features(model, df, df)
        m = X_all.notna().all(axis=1)
        scores = cross_val_score(clone(model), X_all[m], df.loc[m.values, TARGET],
                                 cv=5, scoring="r2")
        results["cv"] = scores
    except Exception:  # noqa: BLE001
        pass
    return results


def model_page():
    df, model = safe_load()
    page_header("Model", "Algorithm and evaluation results")

    final = model.steps[-1][1] if hasattr(model, "steps") else model
    alpha = getattr(final, "alpha", None)
    st.markdown(
        f'<div class="card"><b>Algorithm used: Ridge Regression</b><br>'
        f'Ridge Regression is linear regression with L2 regularisation. It adds a '
        f'penalty on large coefficients, which reduces overfitting and keeps the model '
        f'stable when features are correlated.'
        f'{f"<br><br>Saved model type: <code>{type(final).__name__}</code>, alpha = {alpha}" if alpha is not None else ""}'
        f'</div>', unsafe_allow_html=True)
    st.write("")

    try:
        res = evaluate_model()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not evaluate the model: {e}")
        return

    st.markdown("#### Test Set Results")
    st.caption(f"Computed from the saved model on a held-out {int(TEST_SIZE*100)}% test split "
               f"(random_state={RANDOM_STATE}, {res['n_test']} rows). If your training "
               f"script used different values, change TEST_SIZE / RANDOM_STATE at the top.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("R² Score", f"{res['r2']:.4f}")
    with c2:
        metric_card("Mean Absolute Error", f"{res['mae']:.4f}")
    with c3:
        metric_card("Root Mean Squared Error", f"{res['rmse']:.4f}")

    st.write("")
    st.markdown("#### Cross-Validation Results (5-fold, R²)")
    if res["cv"] is None:
        st.info("Cross-validation could not be computed for this model.")
    else:
        cv = res["cv"]
        st.caption("Separate from the test results above: the data is split 5 ways and "
                   "a fresh copy of the model is fitted on each fold.")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Mean CV R²", f"{cv.mean():.4f}", f"± {cv.std():.4f}")
        with c2:
            fig = px.bar(x=[f"Fold {i+1}" for i in range(len(cv))], y=cv,
                         color_discrete_sequence=[PURPLE], text_auto=".3f")
            fig.update_layout(xaxis_title="", yaxis_title="R²", height=260)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("#### Model Comparison")
    if COMPARISON_PATH.exists():
        try:
            st.dataframe(pd.read_csv(COMPARISON_PATH), use_container_width=True,
                         hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read {COMPARISON_PATH.name}: {e}")
    else:
        st.info("No saved comparison results found (expected `model/model_comparison.csv`), "
                "so none are shown.")


# --------------------------------------------------------------------------
# Navigation (native Streamlit sidebar links - each item is directly clickable)
# --------------------------------------------------------------------------
PAGES = {
    "Home": st.Page(home_page, title="Home", icon=":material/home:", default=True,
                    url_path="home"),
    "Dataset": st.Page(dataset_page, title="Dataset", icon=":material/table_chart:",
                       url_path="dataset"),
    "Analysis": st.Page(analysis_page, title="Analysis", icon=":material/bar_chart:",
                        url_path="analysis"),
    "Prediction": st.Page(prediction_page, title="Prediction",
                          icon=":material/online_prediction:", url_path="prediction"),
    "Model": st.Page(model_page, title="Model", icon=":material/psychology:",
                     url_path="model"),
}

with st.sidebar:
    st.markdown('<div class="brand">🎓 EduInsight<small>Student Analytics</small></div>',
                unsafe_allow_html=True)

st.navigation(list(PAGES.values())).run()