import pytest
from app.core import timeline

UNITS=[{'id':1,'start':0,'end':5,'kind':'TRANSCRIBED','text':'Discussion'}]
VALID='{"segments":[{"first_id":1,"last_id":1,"label":"Content","reason":"Discussion"}],"summary":null}'

def test_complete_json_can_be_fenced_or_surrounded_by_prose():
    for text in [VALID,'```json\n'+VALID+'\n```','Here is the result:\n'+VALID+'\nDone.']:
        rows,_=timeline.parse_response(text,UNITS)
        assert rows[0]['start']==0 and rows[0]['end']==5

@pytest.mark.parametrize('text',['No ads found.','{"refusal":"no"}','[','[null]','{"segments":[]}',VALID[:-1],VALID.replace('"Content"','"unknown"'),VALID.replace('"first_id":1','"first_id":-1'),VALID.replace('"last_id":1','"last_id":NaN'),VALID.replace('"last_id":1','"last_id":Infinity'),VALID+VALID])
def test_invalid_analysis_is_not_a_clean_episode(text):
    with pytest.raises(timeline.TimelineError): timeline.parse_response(text,UNITS)
