"""Thin Qlib adapter used by the evaluator."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable
import warnings

import pandas as pd

from lineageevo_eval.config import DatasetConfig


@contextmanager
def suppress_qlib_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        yield


class QlibDataClient:
    def __init__(self, dataset: DatasetConfig) -> None:
        self.dataset = dataset
        self._initialized = False

    def features(self, fields: Iterable[str], start_time: str, end_time: str) -> pd.DataFrame:
        self._ensure_qlib()
        from qlib.data import D

        with suppress_qlib_warnings():
            return D.features(
                D.instruments(self.dataset.market),
                list(fields),
                start_time=start_time,
                end_time=end_time,
            )

    def _ensure_qlib(self) -> None:
        if self._initialized:
            return
        try:
            import qlib
            from qlib.constant import REG_CN
        except Exception as exc:
            raise RuntimeError(
                "pyqlib is not available. Install it with `pip install -e .[qlib]` "
                "or use an environment that can import qlib."
            ) from exc

        region = REG_CN if self.dataset.region.lower() == "cn" else self.dataset.region
        qlib.init(provider_uri=self.dataset.provider_uri, region=region)
        self._initialized = True
