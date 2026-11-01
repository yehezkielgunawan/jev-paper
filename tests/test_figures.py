from __future__ import annotations

from matplotlib.figure import Figure

from jev_risk.figures import create_evaluation_figures
from jev_risk.predictions import build_prediction_frame


def test_create_evaluation_figures_writes_comparison_plots(tmp_path):
    y_true = [0, 0, 1, 1, 0, 1]
    rf = build_prediction_frame(
        [f"id-{i}" for i in range(6)],
        y_true,
        [0, 0, 1, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.6, 0.7, 0.9],
        [1] * 6,
    )
    jev = build_prediction_frame(
        [f"id-{i}" for i in range(6)],
        y_true,
        [0, 1, 1, 0, 0, 1],
        [0.1, 0.7, 0.9, 0.4, 0.2, 0.8],
        [10] * 6,
        confidence=[0.9, 0.4, 0.8, 0.6, 0.95, 0.85],
    )

    files = create_evaluation_figures({"random_forest": rf, "jev": jev}, tmp_path)

    assert (tmp_path / "confusion_matrix_random_forest.png").is_file()
    assert (tmp_path / "confusion_matrix_jev.png").is_file()
    assert (tmp_path / "roc_curve.png").is_file()
    assert (tmp_path / "precision_recall_curve.png").is_file()
    assert (tmp_path / "calibration_curve.png").is_file()
    assert (tmp_path / "confidence_coverage.png").is_file()
    assert set(files) == {
        "confusion_matrix_random_forest.png",
        "confusion_matrix_jev.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "confidence_coverage.png",
    }


def test_precision_recall_plot_uses_prevalence_line_not_dummy_curve(tmp_path, monkeypatch):
    truth = [0, 0, 1, 0, 1, 0]
    sample_ids = [f"id-{index}" for index in range(len(truth))]
    dummy = build_prediction_frame(sample_ids, truth, [0] * 6, [1 / 3] * 6, [1] * 6)
    rf = build_prediction_frame(
        sample_ids, truth, [0, 0, 1, 0, 1, 0], [0.1, 0.2, 0.7, 0.3, 0.8, 0.4], [1] * 6
    )
    plotted_labels = []
    original_savefig = Figure.savefig

    def capture_precision_recall(figure, path, *args, **kwargs):
        if path.name == "precision_recall_curve.png":
            plotted_labels.extend(line.get_label() for line in figure.axes[0].lines)
        return original_savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_precision_recall)
    create_evaluation_figures({"dummy": dummy, "random_forest": rf}, tmp_path)

    assert plotted_labels == ["random forest", "Positive prevalence"]
