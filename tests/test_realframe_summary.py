import json
from pathlib import Path

from benchmark.realframe_vlm import print_summary


def test_recorded_counts_keep_empty_tables_separate(capsys):
    record = Path(__file__).resolve().parents[1] / "results/recorded/cups2/realframe_vlm.json"
    trials = json.loads(record.read_text())
    print_summary(trials)
    assert capsys.readouterr().out.splitlines() == [
        "Gemini PASS on flat print: 10 / 15",
        "Gemini PASS on real object: 9 / 9",
        "Gemini PASS on empty table: 0 / 3",
    ]

    # Even if the model passes an empty control, it cannot add a print false pass.
    trials["bare table 01"]["gemini_pass"] = True
    print_summary(trials)
    output = capsys.readouterr().out
    assert "Gemini PASS on flat print: 10 / 15" in output
    assert "Gemini PASS on empty table: 1 / 3" in output
