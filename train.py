
# ==========================================================
# STUDENT PERFORMANCE ANALYSIS AND PREDICTION
# MODEL TRAINING AND EVALUATION
# ==========================================================

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COLUMN = "Exam_Score"


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("\n" + "=" * 60)
print("STUDENT PERFORMANCE ANALYSIS AND PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv("data/StudentPerformanceFactors.csv")

df = df.dropna(subset=[TARGET_COLUMN])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("Dataset loaded successfully.")
print("Records:", len(df))
print("Features:", X.shape[1])


# ==========================================================
# 2. IDENTIFY FEATURE TYPES
# ==========================================================

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()


# ==========================================================
# 3. DATA PREPROCESSING
# ==========================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================================
# 4. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================================
# 5. CREATE OUTPUT DIRECTORIES
# ==========================================================

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================================
# 6. DEFINE MODELS
# ==========================================================

models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=10)
}


# ==========================================================
# 7. MODEL TRAINING AND CROSS-VALIDATION
# ==========================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = {}
trained_models = {}

print("\n" + "=" * 60)
print("MODEL TRAINING AND COMPARISON")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "CV R2": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Test R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        )
    }

    results[name] = metrics
    trained_models[name] = pipeline

    print("Cross-validation R2:", round(metrics["CV R2"], 4))
    print("Test R2:", round(metrics["Test R2"], 4))


# Save initial comparison

initial_results = pd.DataFrame(results).T

initial_results.to_csv(
    "outputs/initial_model_comparison.csv"
)


# ==========================================================
# 8. RIDGE REGRESSION OPTIMIZATION
# ==========================================================

print("\nOptimizing Ridge Regression...")

ridge_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge())
])

parameter_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 50, 100]
}

search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=parameter_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("Best Parameters:", search.best_params_)


# ==========================================================
# 9. FINAL MODEL SELECTION
# ==========================================================

all_candidates = {
    **trained_models,
    "Ridge Regression Tuned": best_model
}

candidate_scores = {
    name: (
        search.best_score_
        if name == "Ridge Regression Tuned"
        else results[name]["CV R2"]
    )
    for name in all_candidates
}

best_model_name = max(
    candidate_scores,
    key=candidate_scores.get
)

best_model = all_candidates[best_model_name]

best_predictions = best_model.predict(X_test)

best_metrics = {
    "Model": best_model_name,
    "CV R2": candidate_scores[best_model_name],
    "Test R2": r2_score(y_test, best_predictions),
    "MAE": mean_absolute_error(y_test, best_predictions),
    "RMSE": np.sqrt(
        mean_squared_error(y_test, best_predictions)
    )
}

results["Ridge Regression Tuned"] = {
    "CV R2": search.best_score_,
    "CV Std": search.cv_results_["std_test_score"][
        search.best_index_
    ],
    "Test R2": best_metrics["Test R2"],
    "MAE": best_metrics["MAE"],
    "RMSE": best_metrics["RMSE"]
}


# ==========================================================
# 10. SAVE TRAINED MODEL
# ==========================================================

joblib.dump(
    best_model,
    "model/model.pkl"
)


# ==========================================================
# 11. SAVE MODEL METRICS
# ==========================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="CV R2",
    ascending=False
)

results_df.to_csv(
    "outputs/model_comparison.csv"
)

with open("outputs/model_metrics.json", "w") as file:

    json.dump({
        "best_model": best_model_name,
        "metrics": {
            key: float(value) if isinstance(
                value, (np.integer, np.floating)
            ) else value
            for key, value in best_metrics.items()
        }
    }, file, indent=4)


# ==========================================================
# 12. ACTUAL VS PREDICTED GRAPH
# ==========================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=y_test,
    y=best_predictions,
    alpha=0.6
)

min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Scores")

plt.tight_layout()

plt.savefig("outputs/actual_vs_predicted.png")
plt.close()


# ==========================================================
# 13. RESIDUAL DISTRIBUTION
# ==========================================================

residuals = y_test - best_predictions

plt.figure(figsize=(9, 6))

sns.histplot(
    residuals,
    kde=True,
    bins=30
)

plt.axvline(0, linestyle="--")

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig("outputs/residual_distribution.png")
plt.close()


# ==========================================================
# 14. MODEL COMPARISON GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))

results_df["Test R2"].plot(kind="bar")

plt.ylabel("Test R2 Score")
plt.xlabel("Regression Models")
plt.title("Model Performance Comparison")

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig("outputs/model_comparison.png")
plt.close()


# ==========================================================
# 15. ERROR ANALYSIS
# ==========================================================

error_df = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": best_predictions,
    "Residual": residuals.values,
    "Absolute Error": np.abs(residuals.values)
})

error_df.to_csv(
    "outputs/prediction_errors.csv",
    index=False
)

largest_errors = error_df.sort_values(
    by="Absolute Error",
    ascending=False
).head(10)

print("\nLargest Prediction Errors:")
print(largest_errors.round(3))


# Residual vs predicted graph

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=best_predictions,
    y=residuals,
    alpha=0.6
)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Scores")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Scores")

plt.tight_layout()

plt.savefig("outputs/residual_analysis.png")
plt.close()


# ==========================================================
# 16. FINAL TRAINING SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("              TRAINING SUMMARY")
print("=" * 60)

print("\nDATASET INFORMATION")
print("-" * 60)

print("Total Records       :", len(df))
print("Input Features      :", X.shape[1])
print("Training Records    :", len(X_train))
print("Testing Records     :", len(X_test))

print("\nMODEL PERFORMANCE")
print("-" * 60)

print("Selected Model      :", best_model_name)
print("Cross-Validation R2 :", round(best_metrics["CV R2"], 4))
print("Test R2 Score       :", round(best_metrics["Test R2"], 4))
print("Mean Absolute Error :", round(best_metrics["MAE"], 4))
print("Root Mean Squared Error:", round(best_metrics["RMSE"], 4))

print("\nSAVED FILES")
print("-" * 60)

print("Trained Model       : model/model.pkl")
print("Model Comparison    : outputs/model_comparison.csv")
print("Model Metrics       : outputs/model_metrics.json")
print("Actual vs Predicted : outputs/actual_vs_predicted.png")
print("Residual Distribution: outputs/residual_distribution.png")
print("Residual Analysis   : outputs/residual_analysis.png")
print("Model Comparison Graph: outputs/model_comparison.png")
print("Prediction Errors   : outputs/prediction_errors.csv")

print("\n" + "=" * 60)
print("          MODEL TRAINING COMPLETED")
print("=" * 60)
