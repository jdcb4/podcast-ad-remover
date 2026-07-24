import pytest

from app.core.ai_services import AdDetectionParseError, AdDetector


def test_parse_ad_response_accepts_fenced_json_and_string_times():
    detector = AdDetector()

    result = detector._parse_ad_response("""
```json
[
  {"start": "1.5", "end": "4.25", "label": "Ad", "reason": "Sponsor read"}
]
```
""")

    assert result == [
        {"start": 1.5, "end": 4.25, "label": "Ad", "reason": "Sponsor read"}
    ]


def test_parse_ad_response_accepts_segments_wrapper_and_discards_bad_rows():
    detector = AdDetector()

    result = detector._parse_ad_response("""
Here is the answer:
{"segments": [
  {"start": 5, "end": 3, "label": "Ad"},
  {"start": 10, "end": 15, "label": "Promo"}
]}
""")

    assert result == [
        {"start": 10.0, "end": 15.0, "label": "Promo", "reason": ""}
    ]


def test_parse_ad_response_rejects_non_json_instead_of_treating_it_as_no_ads():
    detector = AdDetector()

    with pytest.raises(AdDetectionParseError):
        detector._parse_ad_response("No ads found.")


def test_parse_ad_response_accepts_explicit_empty_json_array():
    detector = AdDetector()

    assert detector._parse_ad_response("[]") == []
