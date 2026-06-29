import csv
from pathlib import Path

import pandas as pd

from lineageevo_eval.config import EvalConfig
from lineageevo_eval.engine import EvaluationOptions, _select_factors, evaluate_file


def test_select_factors_uses_input_order_and_keeps_provided_direction():
    rows = [
        {"factor_id": "a", "status": "ok", "qlib_expression": "$close", "orientation": -1},
        {"factor_id": "b", "status": "ok", "qlib_expression": "$open"},
        {"factor_id": "c", "status": "ok", "qlib_expression": "$low"},
    ]

    selected = _select_factors(rows, 2)

    assert [row["factor_id"] for row in selected] == ["a", "b"]
    assert selected[0]["orientation"] == -1
    assert selected[1]["orientation"] == 1
    assert "selection_score" not in selected[0]


def test_select_factors_does_not_backfill_failed_topk_rows():
    rows = [
        {"factor_id": "a", "status": "failed", "qlib_expression": None},
        {"factor_id": "b", "status": "ok", "qlib_expression": "$open"},
        {"factor_id": "c", "status": "ok", "qlib_expression": "$low"},
    ]

    selected = _select_factors(rows, 2)

    assert [row["factor_id"] for row in selected] == ["b"]


class FakeClient:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def features(self, fields, start_time, end_time):
        dates = pd.date_range(start_time, end_time, freq="D")
        index = pd.MultiIndex.from_product([dates, ["a", "b", "c"]], names=["datetime", "instrument"])
        data = pd.DataFrame(index=index)
        for field in fields:
            if field == self.config.evaluation.label_expression:
                data[field] = [1.0, 2.0, 3.0] * len(dates)
            elif "Rank($close" in field or field == "$close":
                data[field] = [1.0, 2.0, 3.0] * len(dates)
            elif "Mean($close" in field:
                data[field] = [3.0, 2.0, 1.0] * len(dates)
            else:
                data[field] = [2.0, 3.0, 1.0] * len(dates)
        return data


def write_config(tmp_path: Path) -> Path:
    provider = tmp_path / "qlib_data"
    provider.mkdir()
    config = tmp_path / "eval.local.toml"
    config.write_text(
        f"""
default_dataset = "csi500"
output_root = "runs"

[datasets.csi500]
provider_uri = "{provider.as_posix()}"
region = "cn"
market = "csi500"
benchmark = "SH000905"

[datasets.sp500]
provider_uri = "{provider.as_posix()}"
region = "us"
market = "sp500"
benchmark = "SPY"

[time_split]
train_start = "2020-01-01"
train_end = "2020-01-03"
valid_start = "2020-01-04"
valid_end = "2020-01-06"
test_start = "2020-01-07"
test_end = "2020-01-09"
""",
        encoding="utf-8",
    )
    return config


def test_evaluate_file_writes_expected_outputs_with_fake_client(tmp_path):
    input_path = tmp_path / "factors.jsonl"
    input_path.write_text(
        '{"factor_id": "positive", "expression": "Rank($close)", "baseline": "mock"}\n'
        '{"factor_id": "negative", "expression": "TsMean($close, 5)", "baseline": "mock"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = evaluate_file(
        input_path,
        EvaluationOptions(
            config_path=write_config(tmp_path),
            select="top1",
            selection_metric="valid_ic",
            output_dir=output_dir,
            run_backtest=False,
        ),
        data_client_factory=FakeClient,
    )

    assert result.summary["selected_count"] == 1
    assert result.summary["evaluation_mode"] == "test_only"
    assert (output_dir / "factor_evaluations.csv").exists()
    assert (output_dir / "selected_factors.csv").exists()
    assert (output_dir / "composite_test_ic_results.csv").exists()
    selected = list(csv.DictReader((output_dir / "selected_factors.csv").open(encoding="utf-8")))
    assert selected[0]["factor_id"] == "positive"
    assert selected[0]["orientation"] == "1"
    assert "train_ic" not in selected[0]
    assert "valid_ic" not in selected[0]
    backtest = list(csv.DictReader((output_dir / "backtest_summary.csv").open(encoding="utf-8")))
    assert backtest[0]["status"] == "not_run"
