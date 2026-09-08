import pytest
from app.core.ai_services import AdDetector, AnalysisError


def test_valid_empty_and_wrapped_segments():
    detector = AdDetector()
    assert detector._parse_ad_response('[]') == []
    assert detector._parse_ad_response('```json\n{"segments":[{"start":"1.5","end":4,"label":"ad"}]}\n```') == [
        {"start":1.5,"end":4.0,"label":"Ad","reason":""}]


@pytest.mark.parametrize('text', ['No ads found.', '{"refusal":"no"}', '[', '[null]',
    '[{"start":5,"end":3}]', '[{"start":0,"end":NaN}]', '[{"start":0,"end":Infinity}]',
    '[{"start":0,"end":3,"label":"unknown"}]', '[{"start":-1,"end":3}]'])
def test_invalid_analysis_is_not_a_clean_episode(text):
    with pytest.raises(AnalysisError):
        AdDetector()._parse_ad_response(text)
