"""Leakage-safe preprocessing and supervised model pipelines."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from jev_risk.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from jev_risk.predictions import build_prediction_frame
from jev_risk.split import DatasetSplit


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing fitted only inside a training pipeline."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_model_pipelines(y_train: Any) -> dict[str, Pipeline]:
    """Build baseline estimators with their own fitted preprocessing pipelines."""
    positives = int(np.asarray(y_train).sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both classes")

    def pipeline(estimator: Any) -> Pipeline:
        return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])

    return {
        "dummy": pipeline(DummyClassifier(strategy="prior")),
        "random_forest": pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "xgboost": pipeline(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=negatives / positives,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            )
        ),
    }


def hyperparameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Return the bounded training-only searches specified by the experiment."""
    return {
        "random_forest": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "xgboost": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.03, 0.1],
        },
    }


def fit_classical_models(
    split: DatasetSplit,
    tune: bool = True,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Fit the baseline, Random Forest, and XGBoost using training data only."""
    models = build_model_pipelines(split.y_train)
    metadata: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        if tune and name in {"random_forest", "xgboost"}:
            positives = int(split.y_train.sum())
            negatives = int(len(split.y_train) - positives)
            if min(positives, negatives) < 5:
                raise ValueError(
                    "Five-fold stratified tuning requires at least 5 samples per class"
                )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(
                estimator=model,
                param_grid=hyperparameter_grids()[name],
                scoring="average_precision",
                cv=cv,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(split.X_train, split.y_train)
            models[name] = search.best_estimator_
            metadata[name] = {
                "tuned": True,
                "selection_metric": "average_precision",
                "best_cv_score": float(search.best_score_),
                "best_params": search.best_params_,
                "cv_folds": 5,
            }
        else:
            model.fit(split.X_train, split.y_train)
            metadata[name] = {"tuned": False, "best_params": {}}
    return models, metadata


def predict_model(name: str, model: Any, split: DatasetSplit):
    """Generate per-record local inference timings on the frozen test rows."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"{name} model does not contain positive class 1")
    positive_index = classes.index(1)
    probabilities: list[float] = []
    predicted: list[int] = []
    latencies: list[float] = []
    for index in range(len(split.X_test)):
        row = split.X_test.iloc[[index]]
        start = time.perf_counter()
        probability = float(model.predict_proba(row)[0, positive_index])
        latencies.append((time.perf_counter() - start) * 1000)
        probabilities.append(probability)
        predicted.append(int(probability > 0.5))
    return build_prediction_frame(
        sample_ids=split.test_ids,
        y_true=split.y_test.tolist(),
        y_pred=predicted,
        y_prob=probabilities,
        latency_ms=latencies,
    )
