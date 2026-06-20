from pathlib import Path

import pytest

from lineageevo_eval.config import ConfigError, load_config, validate_provider_uri


def write_config(tmp_path: Path) -> Path:
    csi = tmp_path / "cn_data"
    sp = tmp_path / "us_data"
    csi.mkdir()
    sp.mkdir()
    config = tmp_path / "eval.local.toml"
    config.write_text(
        f"""
default_dataset = "csi500"
output_root = "runs"

[datasets.csi500]
provider_uri = "{csi.as_posix()}"
region = "cn"
market = "csi500"
benchmark = "SH000905"

[datasets.sp500]
provider_uri = "{sp.as_posix()}"
region = "us"
market = "sp500"
benchmark = "SPY"
""",
        encoding="utf-8",
    )
    return config


def test_load_config_defaults_to_csi500(tmp_path):
    config = load_config(write_config(tmp_path))

    assert config.dataset.name == "csi500"
    assert config.dataset.market == "csi500"
    assert config.time_split.train_start == "2015-01-01"


def test_load_config_switches_to_sp500(tmp_path):
    config = load_config(write_config(tmp_path), "sp500")

    assert config.dataset.name == "sp500"
    assert config.dataset.region == "us"
    assert config.dataset.benchmark == "SPY"


def test_validate_provider_uri_reports_missing_path(tmp_path):
    config_path = tmp_path / "eval.local.toml"
    missing = tmp_path / "missing"
    config_path.write_text(
        f"""
[datasets.csi500]
provider_uri = "{missing.as_posix()}"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(ConfigError, match="does not exist"):
        validate_provider_uri(config.dataset)
