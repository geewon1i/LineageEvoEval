"""End-to-end evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import json

import pandas as pd

from lineageevo_eval.config import EvalConfig, load_config, validate_provider_uri
from lineageevo_eval.dsl import FactorExpression, QlibExpressionNormalizer, validate_expression_length
from lineageevo_eval.io import FactorRecord, load_factor_records, write_csv, write_json
from lineageevo_eval.metrics import (
    clean_factor_frame,
    daily_zscore,
    flatten_risk_analysis,
    ic_metrics,
    max_drawdown,
)
from lineageevo_eval.qlib_backend import QlibDataClient


TOPK_CHOICES = {"top1": 1, "top5": 5, "top20": 20}
SELECTION_METRICS = {"valid_ic", "train_ic"}


@dataclass(frozen=True)
class EvaluationOptions:
    config_path: str | Path = "configs/eval.local.toml"
    dataset: str | None = None
    select: str | None = None
    selection_metric: str | None = None
    output_dir: str | Path | None = None
    run_backtest: bool | None = None


@dataclass(frozen=True)
class EvaluationRunResult:
    output_dir: str
    summary: dict[str, Any]


DataClientFactory = Callable[[EvalConfig], Any]


def evaluate_file(
    input_path: str | Path,
    options: EvaluationOptions | None = None,
    *,
    data_client_factory: DataClientFactory | None = None,
) -> EvaluationRunResult:
    options = options or EvaluationOptions()
    config = load_config(options.config_path, options.dataset)
    validate_provider_uri(config.dataset)

    select = options.select or config.selection.default_select
    selection_metric = options.selection_metric or config.selection.default_selection_metric
    _validate_options(select, selection_metric)
    run_backtest = config.backtest.enabled if options.run_backtest is None else options.run_backtest

    records = load_factor_records(input_path)
    out_dir = _resolve_output_dir(config, options.output_dir, input_path, config.dataset.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = data_client_factory(config) if data_client_factory else QlibDataClient(config.dataset)
    normalizer = QlibExpressionNormalizer()

    config_snapshot = {
        **config.as_dict(),
        "input_path": str(Path(input_path).resolve()),
        "select": select,
        "selection_metric": selection_metric,
        "evaluation_mode": "test_only",
        "orientation_policy": "as_provided",
        "run_backtest": run_backtest,
    }
    write_json(out_dir / "config_snapshot.json", config_snapshot)

    factor_rows = _evaluate_records(records, config, client, normalizer)
    selected_rows = _select_factors(factor_rows, TOPK_CHOICES[select])
    test_rows = _test_ic_rows(selected_rows, config, client)
    composite_rows, signal = _composite_test_rows(selected_rows, config, client)
    backtest_rows, daily_rows = _backtest_rows(signal, selected_rows, config, run_backtest)

    write_csv(out_dir / "factor_evaluations.csv", factor_rows)
    write_csv(out_dir / "selected_factors.csv", selected_rows)
    write_csv(out_dir / "test_ic_results.csv", test_rows)
    write_csv(out_dir / "composite_test_ic_results.csv", composite_rows)
    write_csv(out_dir / "backtest_summary.csv", backtest_rows)
    write_csv(out_dir / "backtest_daily_report.csv", daily_rows)

    summary = _run_summary(
        config=config,
        input_path=input_path,
        output_dir=out_dir,
        select=select,
        selection_metric=selection_metric,
        factor_rows=factor_rows,
        selected_rows=selected_rows,
        composite_rows=composite_rows,
        backtest_rows=backtest_rows,
    )
    write_json(out_dir / "run_summary.json", summary)
    return EvaluationRunResult(output_dir=str(out_dir), summary=summary)


def _validate_options(select: str, selection_metric: str) -> None:
    if select not in TOPK_CHOICES:
        raise ValueError(f"select must be one of {', '.join(TOPK_CHOICES)}")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {', '.join(sorted(SELECTION_METRICS))}")


def _resolve_output_dir(config: EvalConfig, output_dir: str | Path | None, input_path: str | Path, dataset: str) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    config_dir = Path(config.config_path).parent
    root = Path(config.output_root)
    if not root.is_absolute():
        root = (config_dir.parent / root).resolve()
    stem = Path(input_path).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{stamp}_{dataset}_{stem}"


def _evaluate_records(records: list[FactorRecord], config: EvalConfig, client: Any, normalizer: QlibExpressionNormalizer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        base = _base_row(record)
        try:
            expression = FactorExpression(record.expression)
            validate_expression_length(expression, config.evaluation.factor_length_limit)
            qlib_expression = normalizer.normalize(expression)
            rows.append(
                {
                    **base,
                    "qlib_expression": qlib_expression,
                    "status": "ok",
                    "failure_reason": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "qlib_expression": None,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _base_row(record: FactorRecord) -> dict[str, Any]:
    return {
        "input_index": record.input_index,
        "factor_id": record.factor_id,
        **record.metadata,
        "expression": record.expression,
    }


def _load_factor_label(client: Any, qlib_expression: str, label_expression: str, start: str, end: str) -> pd.DataFrame:
    data = client.features([qlib_expression, label_expression], start_time=start, end_time=end)
    data = data.rename(columns={qlib_expression: "factor", label_expression: "label"})
    data = clean_factor_frame(data)
    if data.empty:
        raise ValueError("Qlib produced empty factor/label data")
    return data


def _select_factors(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows[:top_k]:
        if row.get("status") == "ok":
            selected.append(
                {
                    **row,
                    "selection_rank": len(selected) + 1,
                    "orientation": 1,
                }
            )
    return selected


def _test_ic_rows(selected_rows: list[dict[str, Any]], config: EvalConfig, client: Any) -> list[dict[str, Any]]:
    rows = []
    for row in selected_rows:
        try:
            data = _load_factor_label(
                client,
                str(row["qlib_expression"]),
                config.evaluation.label_expression,
                config.time_split.test_start,
                config.time_split.test_end,
            )
            data["factor"] = data["factor"] * int(row["orientation"])
            test_ic, test_icir = ic_metrics(data, config.evaluation.ic_method)
            rows.append(
                {
                    "selection_rank": row["selection_rank"],
                    "factor_id": row["factor_id"],
                    "expression": row["expression"],
                    "test_ic": test_ic,
                    "test_icir": test_icir,
                    "status": "ok",
                    "failure_reason": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "selection_rank": row["selection_rank"],
                    "factor_id": row["factor_id"],
                    "expression": row["expression"],
                    "test_ic": None,
                    "test_icir": None,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _composite_test_rows(
    selected_rows: list[dict[str, Any]],
    config: EvalConfig,
    client: Any,
) -> tuple[list[dict[str, Any]], pd.Series | None]:
    if not selected_rows:
        return [
            {
                "signal_name": "oriented_equal_weight_top_k",
                "selected_count": 0,
                "factor_ids": "[]",
                "test_ic": None,
                "test_icir": None,
                "status": "failed",
                "failure_reason": "no selected factors",
            }
        ], None
    try:
        signal = _composite_signal(selected_rows, config, client)
        label = _load_label(client, config)
        data = signal.to_frame("factor").join(label, how="inner").dropna()
        if data.empty:
            raise ValueError("composite factor/label data is empty")
        test_ic, test_icir = ic_metrics(data, config.evaluation.ic_method)
        return [
            {
                "signal_name": "oriented_equal_weight_top_k",
                "selected_count": len(selected_rows),
                "factor_ids": json.dumps([row["factor_id"] for row in selected_rows], ensure_ascii=False),
                "test_ic": test_ic,
                "test_icir": test_icir,
                "status": "ok",
                "failure_reason": None,
            }
        ], signal
    except Exception as exc:
        return [
            {
                "signal_name": "oriented_equal_weight_top_k",
                "selected_count": len(selected_rows),
                "factor_ids": json.dumps([row["factor_id"] for row in selected_rows], ensure_ascii=False),
                "test_ic": None,
                "test_icir": None,
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        ], None


def _composite_signal(selected_rows: list[dict[str, Any]], config: EvalConfig, client: Any) -> pd.Series:
    fields = list(dict.fromkeys(str(row["qlib_expression"]) for row in selected_rows))
    data = client.features(fields, start_time=config.time_split.test_start, end_time=config.time_split.test_end)
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    oriented = pd.DataFrame(index=data.index)
    for row in selected_rows:
        oriented[str(row["factor_id"])] = data[str(row["qlib_expression"])] * int(row["orientation"])
    standardized = oriented.groupby(level="datetime", group_keys=False).apply(daily_zscore)
    signal = standardized.mean(axis=1).dropna()
    if signal.empty:
        raise ValueError("composite signal is empty")
    return pd.Series(signal, name="score")


def _load_label(client: Any, config: EvalConfig) -> pd.DataFrame:
    label_expression = config.evaluation.label_expression
    data = client.features([label_expression], start_time=config.time_split.test_start, end_time=config.time_split.test_end)
    data = data.rename(columns={label_expression: "label"})
    data = clean_factor_frame(data)
    if data.empty:
        raise ValueError("test label data is empty")
    return data


def _backtest_rows(
    signal: pd.Series | None,
    selected_rows: list[dict[str, Any]],
    config: EvalConfig,
    run_backtest: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not run_backtest:
        return [
            {
                "status": "not_run",
                "reason": "disabled by --no-backtest or config",
                "selected_count": len(selected_rows),
                "benchmark": config.dataset.benchmark,
            }
        ], []
    if signal is None or not selected_rows:
        return [
            {
                "status": "failed",
                "reason": "no composite signal available",
                "selected_count": len(selected_rows),
                "benchmark": config.dataset.benchmark,
            }
        ], []

    try:
        from qlib.backtest import backtest
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        strategy = TopkDropoutStrategy(
            signal=signal,
            topk=config.backtest.topk,
            n_drop=config.backtest.n_drop,
            risk_degree=config.backtest.risk_degree,
        )
        executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
        portfolio_metric, _indicator_metric = backtest(
            start_time=config.time_split.test_start,
            end_time=config.time_split.test_end,
            strategy=strategy,
            executor=executor,
            benchmark=config.dataset.benchmark,
            account=config.backtest.account,
        )
        report = _extract_report(portfolio_metric)
        daily_rows = report.reset_index().to_dict(orient="records") if hasattr(report, "reset_index") else []
        risk = risk_analysis(report["return"], freq="day") if "return" in report else {}
        summary = flatten_risk_analysis(risk)
        summary.update(
            {
                "status": "ok",
                "selected_count": len(selected_rows),
                "benchmark": config.dataset.benchmark,
            }
        )
        if "max_drawdown" not in summary and "return" in report:
            summary["max_drawdown"] = max_drawdown(report["return"])
        return [summary], daily_rows
    except Exception as exc:
        return [
            {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "selected_count": len(selected_rows),
                "benchmark": config.dataset.benchmark,
            }
        ], []


def _extract_report(portfolio_metric: Any) -> Any:
    if isinstance(portfolio_metric, tuple):
        return portfolio_metric[0]
    if isinstance(portfolio_metric, dict) and "1day" in portfolio_metric:
        value = portfolio_metric["1day"]
        return value[0] if isinstance(value, tuple) else value
    return portfolio_metric


def _run_summary(
    *,
    config: EvalConfig,
    input_path: str | Path,
    output_dir: Path,
    select: str,
    selection_metric: str,
    factor_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    composite_rows: list[dict[str, Any]],
    backtest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_count = sum(1 for row in factor_rows if row.get("status") == "ok")
    invalid_count = len(factor_rows) - valid_count
    failure_reasons: dict[str, int] = {}
    for row in factor_rows:
        reason = row.get("failure_reason")
        if reason:
            failure_reasons[str(reason)] = failure_reasons.get(str(reason), 0) + 1
    return {
        "dataset": config.dataset.name,
        "market": config.dataset.market,
        "benchmark": config.dataset.benchmark,
        "input_path": str(Path(input_path).resolve()),
        "output_dir": str(output_dir),
        "select": select,
        "selection_metric": selection_metric,
        "evaluation_mode": "test_only",
        "orientation_policy": "as_provided",
        "input_count": len(factor_rows),
        "valid_expression_count": valid_count,
        "invalid_expression_count": invalid_count,
        "selected_count": len(selected_rows),
        "composite_status": composite_rows[0].get("status") if composite_rows else "missing",
        "composite_test_ic": composite_rows[0].get("test_ic") if composite_rows else None,
        "composite_test_icir": composite_rows[0].get("test_icir") if composite_rows else None,
        "backtest_status": backtest_rows[0].get("status") if backtest_rows else "missing",
        "failure_reasons": failure_reasons,
        "files": {
            "config_snapshot": "config_snapshot.json",
            "factor_evaluations": "factor_evaluations.csv",
            "selected_factors": "selected_factors.csv",
            "test_ic_results": "test_ic_results.csv",
            "composite_test_ic_results": "composite_test_ic_results.csv",
            "backtest_summary": "backtest_summary.csv",
            "backtest_daily_report": "backtest_daily_report.csv",
            "run_summary": "run_summary.json",
        },
    }
