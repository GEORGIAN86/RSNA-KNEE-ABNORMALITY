import pandas as pd
import pytest

from Training.ensemble import blend_predictions


def test_probability_blend_aligns_by_uid():
    knee = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "ACL": [0.2, 0.8]})
    sam = pd.DataFrame({"StudyInstanceUID": ["b", "a"], "ACL": [0.4, 0.6]})

    result = blend_predictions(
        knee,
        sam,
        ["ACL"],
        sam_weight=0.25,
        kind="probability",
    )

    assert result.loc[result.StudyInstanceUID == "a", "ACL"].item() == pytest.approx(0.3)


def test_blend_rejects_uid_mismatch():
    knee = pd.DataFrame({"StudyInstanceUID": ["a"], "ACL": [0.2]})
    sam = pd.DataFrame({"StudyInstanceUID": ["b"], "ACL": [0.6]})

    with pytest.raises(ValueError, match="UID"):
        blend_predictions(knee, sam, ["ACL"], sam_weight=0.2, kind="rank")
