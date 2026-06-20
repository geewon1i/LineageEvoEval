from lineageevo_eval.io import load_factor_records


def test_load_factor_records_generates_ids_and_keeps_metadata(tmp_path):
    path = tmp_path / "factors.jsonl"
    path.write_text(
        '{"expression": "Rank($close)", "baseline": "a"}\n'
        '{"factor_id": "custom", "expression": "TsMean($close, 5)", "author": "u"}\n',
        encoding="utf-8",
    )

    records = load_factor_records(path)

    assert records[0].factor_id == "factor_001"
    assert records[0].metadata["baseline"] == "a"
    assert records[1].factor_id == "custom"
    assert records[1].metadata["author"] == "u"
