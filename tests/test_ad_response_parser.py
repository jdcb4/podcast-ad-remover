from app.core.ai_services import AdDetector


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


def test_parse_ad_response_returns_empty_list_for_non_json():
    detector = AdDetector()

    assert detector._parse_ad_response("No ads found.") == []
