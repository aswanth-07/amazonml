from ber.audit import run_audit


def test_audit_on_synthetic(synthetic_dir):
    report = run_audit(synthetic_dir)
    train, test = report["train"], report["test"]

    gt = train["ground_truth"]
    assert gt["pool_records_with_multiple_s1"] == 0
    assert gt["unknown_matched_ids"] == 0
    assert gt["true_pairs_same_country_rate"] == 1.0
    assert 0 < gt["singleton_rate"] < 1
    assert any("exclusive=true" in r for r in train["recommendations"])
    assert any("partition_by_country=true" in r for r in train["recommendations"])

    files = train["files"]
    assert set(files) == {
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    }
    assert all(len(f["tabs_per_line"]) == 1 for f in files.values())
    assert "France" in test["sources"]["s1"]["countries"]
