"""Configuration loading for the portable evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    """Raised when a configuration file is missing or inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    provider_uri: str
    region: str
    market: str
    benchmark: str


@dataclass(frozen=True)
class TimeSplitConfig:
    train_start: str = "2015-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2022-04-30"
    test_start: str = "2022-05-01"
    test_end: str = "2026-04-30"


@dataclass(frozen=True)
class EvaluationConfig:
    label_expression: str = "Ref($close, -2) / Ref($close, -1) - 1"
    ic_method: str = "spearman"
    factor_length_limit: int = 50


@dataclass(frozen=True)
class SelectionDefaults:
    default_select: str = "top1"
    default_selection_metric: str = "valid_ic"


@dataclass(frozen=True)
class BacktestConfig:
    enabled: bool = True
    account: float = 100000000.0
    topk: int = 50
    n_drop: int = 5
    risk_degree: float = 0.95


@dataclass(frozen=True)
class EvalConfig:
    config_path: str
    default_dataset: str
    dataset: DatasetConfig
    output_root: str
    time_split: TimeSplitConfig
    evaluation: EvaluationConfig
    selection: SelectionDefaults
    backtest: BacktestConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, dataset_name: str | None = None) -> EvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(
            f"configuration file not found: {config_path}. "
            "Copy configs/eval.example.toml to configs/eval.local.toml and fill in provider_uri."
        )
    data = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    default_dataset = str(data.get("default_dataset", "csi500"))
    selected_dataset = dataset_name or default_dataset

    datasets = data.get("datasets", {})
    if selected_dataset not in datasets:
        available = ", ".join(sorted(datasets)) or "<none>"
        raise ConfigError(f"dataset '{selected_dataset}' is not configured. Available datasets: {available}")

    dataset = _dataset_config(selected_dataset, datasets[selected_dataset], config_path)
    return EvalConfig(
        config_path=str(config_path.resolve()),
        default_dataset=default_dataset,
        dataset=dataset,
        output_root=str(data.get("output_root", "runs")),
        time_split=_time_split_config(data.get("time_split", {})),
        evaluation=_evaluation_config(data.get("evaluation", {})),
        selection=_selection_defaults(data.get("selection", {})),
        backtest=_backtest_config(data.get("backtest", {})),
    )


def validate_provider_uri(dataset: DatasetConfig) -> None:
    if not dataset.provider_uri:
        raise ConfigError(
            f"provider_uri for dataset '{dataset.name}' is empty. "
            "Please fill it in configs/eval.local.toml."
        )
    provider_path = Path(dataset.provider_uri)
    if not provider_path.exists():
        raise ConfigError(
            f"provider_uri for dataset '{dataset.name}' does not exist: {dataset.provider_uri}. "
            "Please check configs/eval.local.toml."
        )


def _dataset_config(name: str, data: dict[str, Any], config_path: Path) -> DatasetConfig:
    provider_uri = str(data.get("provider_uri", "")).strip()
    if provider_uri:
        provider_path = Path(provider_uri).expanduser()
        if not provider_path.is_absolute():
            provider_path = (config_path.parent / provider_path).resolve()
        provider_uri = str(provider_path)
    return DatasetConfig(
        name=name,
        provider_uri=provider_uri,
        region=str(data.get("region", "cn" if name == "csi500" else "us")),
        market=str(data.get("market", name)),
        benchmark=str(data.get("benchmark", "SH000905" if name == "csi500" else "SPY")),
    )


def _time_split_config(data: dict[str, Any]) -> TimeSplitConfig:
    return TimeSplitConfig(
        train_start=str(data.get("train_start", TimeSplitConfig.train_start)),
        train_end=str(data.get("train_end", TimeSplitConfig.train_end)),
        valid_start=str(data.get("valid_start", TimeSplitConfig.valid_start)),
        valid_end=str(data.get("valid_end", TimeSplitConfig.valid_end)),
        test_start=str(data.get("test_start", TimeSplitConfig.test_start)),
        test_end=str(data.get("test_end", TimeSplitConfig.test_end)),
    )


def _evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    return EvaluationConfig(
        label_expression=str(data.get("label_expression", EvaluationConfig.label_expression)),
        ic_method=str(data.get("ic_method", EvaluationConfig.ic_method)),
        factor_length_limit=int(data.get("factor_length_limit", EvaluationConfig.factor_length_limit)),
    )


def _selection_defaults(data: dict[str, Any]) -> SelectionDefaults:
    return SelectionDefaults(
        default_select=str(data.get("default_select", SelectionDefaults.default_select)),
        default_selection_metric=str(data.get("default_selection_metric", SelectionDefaults.default_selection_metric)),
    )


def _backtest_config(data: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(
        enabled=bool(data.get("enabled", BacktestConfig.enabled)),
        account=float(data.get("account", BacktestConfig.account)),
        topk=int(data.get("topk", BacktestConfig.topk)),
        n_drop=int(data.get("n_drop", BacktestConfig.n_drop)),
        risk_degree=float(data.get("risk_degree", BacktestConfig.risk_degree)),
    )
