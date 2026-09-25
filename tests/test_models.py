from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from jev_risk.models import build_preprocessor, fit_classical_models, predict_model
from jev_risk.split import create_split


def test_preprocessor_fits_training_features_and_encodes_categories(sample_transactions):
    split = create_split(sample_transactions, random_state=9)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(split.X_train, split.y_train)

    assert transformed.shape[0] == len(split.y_train)
    assert transformed.shape[1] > split.X_train.shape[1]


def test_model_factories_build_random_forest_and_xgboost_pipelines(sample_transactions):
    split = create_split(sample_transactions, random_state=9)
    from jev_risk.models import build_model_pipelines

    models = build_model_pipelines(split.y_train)

    assert set(models) == {"dummy", "random_forest", "xgboost"}
    assert all("preprocessor" in model.named_steps for model in models.values())
    assert models["random_forest"].named_steps["model"].class_weight == "balanced"
    assert models["xgboost"].named_steps["model"].scale_pos_weight > 1


def test_fit_classical_models_records_fitted_baselines(sample_transactions):
    split = create_split(sample_transactions, random_state=9)

    models, metadata = fit_classical_models(split, tune=False)

    assert set(models) == {"dummy", "random_forest", "xgboost"}
    assert all(hasattr(model, "predict_proba") for model in models.values())
    assert metadata["random_forest"]["tuned"] is False


def test_tuning_uses_five_fold_average_precision_search(sample_transactions, monkeypatch):
    import jev_risk.models as models_module

    split = create_split(sample_transactions, random_state=9)
    monkeypatch.setattr(
        models_module,
        "hyperparameter_grids",
        lambda: {
            "random_forest": {"model__n_estimators": [2]},
            "xgboost": {"model__n_estimators": [2]},
        },
    )

    _, metadata = models_module.fit_classical_models(split, tune=True)

    for model_name in ("random_forest", "xgboost"):
        assert metadata[model_name]["selection_metric"] == "average_precision"
        assert metadata[model_name]["cv_folds"] == 5
        assert 0 <= metadata[model_name]["best_cv_score"] <= 1


def test_predict_model_exports_probabilities_for_shared_test_ids(sample_transactions):
    split = create_split(sample_transactions, random_state=9)
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    model.fit(split.X_train, split.y_train)

    predictions = predict_model("dummy", model, split)

    assert predictions["sample_id"].tolist() == split.test_ids
    assert predictions["y_true"].tolist() == split.y_test.tolist()
    assert predictions["y_prob"].between(0, 1).all()
    assert predictions["latency_ms"].ge(0).all()
