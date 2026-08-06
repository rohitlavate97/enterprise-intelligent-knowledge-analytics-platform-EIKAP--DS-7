import pandas as pd
from data_pipeline.cleaning.duplicate_detector import DuplicateDetector
from data_pipeline.cleaning.missing_handler import MissingValueHandler
from data_pipeline.cleaning.outlier_detector import OutlierDetector


def test_duplicate_detection():
    detector = DuplicateDetector()
    df = pd.DataFrame({"a": [1, 1, 2]})
    dups = detector.find_duplicates(df)
    assert len(dups) == 1


def test_duplicate_removal():
    detector = DuplicateDetector()
    df = pd.DataFrame({"a": [1, 1, 2]})
    res = detector.remove_duplicates(df)
    assert len(res) == 2


def test_missing_analysis():
    handler = MissingValueHandler()
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    analysis = handler.analyze_missing(df)
    assert analysis["total_missing_cells"] == 1


def test_missing_fill_numeric():
    handler = MissingValueHandler()
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    res = handler.fill_numeric(df, strategy='mean')
    assert res["a"].isna().sum() == 0


def test_missing_fill_categorical():
    handler = MissingValueHandler()
    df = pd.DataFrame({"a": ["x", None, "x"]})
    res = handler.fill_categorical(df, strategy='mode')
    assert res["a"].isna().sum() == 0


def test_smart_fill():
    handler = MissingValueHandler()
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "x"]})
    res = handler.smart_fill(df)
    assert res.isna().sum().sum() == 0


def test_outlier_detection_iqr():
    detector = OutlierDetector()
    df = pd.DataFrame({"a": [1, 2, 3, 100]})
    res = detector.detect_iqr(df)
    assert "a" in res
    assert res["a"]["outlier_count"] == 1


def test_outlier_detection_zscore():
    detector = OutlierDetector()
    df = pd.DataFrame({"a": [1, 2, 3, 100]})
    res = detector.detect_zscore(df, threshold=1.0)
    assert "a" in res
    assert res["a"]["outlier_count"] >= 1


def test_outlier_capping():
    detector = OutlierDetector()
    df = pd.DataFrame({"a": [1, 2, 3, 100]})
    res = detector.cap_outliers(df, method='iqr', factor=1.5)
    assert res["a"].max() < 100
