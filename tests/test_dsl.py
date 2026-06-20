import pytest

from lineageevo_eval.dsl import FactorExpression, QlibExpressionNormalizer


def test_normalizes_lineageevo_dsl_to_qlib():
    normalizer = QlibExpressionNormalizer()

    assert normalizer.normalize(FactorExpression("Rank($close)")) == "Rank($close, 5)"
    assert normalizer.normalize(FactorExpression("TsMean($close, 5)")) == "Mean($close, 5)"
    assert normalizer.normalize(FactorExpression("TsCorr($close, $volume, 20)")) == "Corr($close, $volume, 20)"
    assert (
        normalizer.normalize(FactorExpression("Div(Sub($open, $close), Add(Sub($high, $low), 0.001))"))
        == "(($open - $close) / (($high - $low) + 0.001))"
    )


def test_rejects_unknown_feature_and_constant():
    normalizer = QlibExpressionNormalizer()

    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Rank($foo)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("TsMean($close, 7)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Add($close, 0.1)"))
